"""Trainer for teddyMPNN fine-tuning.

Supports single-GPU and multi-GPU (DDP) training with mixed precision,
gradient checkpointing, and Noam LR scheduling.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from teddympnn.data.features import identify_interface_residues
from teddympnn.models import LigandMPNN, ProteinMPNN
from teddympnn.training.loss import LabelSmoothedNLLLoss
from teddympnn.training.scheduler import NoamScheduler
from teddympnn.weights.io import load_checkpoint_bundle, load_model_weights, save_checkpoint_bundle

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from teddympnn.config import TrainingConfig

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Training loop for ProteinMPNN/LigandMPNN fine-tuning.

    Handles single-GPU and DDP training, mixed precision, gradient
    clipping, checkpointing, and validation.

    Args:
        config: Training configuration.
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        device: Device to train on.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: ProteinMPNN | LigandMPNN,
        train_loader: DataLoader[dict[str, Any]] | Any,
        val_loader: DataLoader[dict[str, Any]] | Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.world_size > 1
        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            model.use_gradient_checkpointing = True

        # Set structure noise from config
        model.augment_eps = config.structure_noise
        self.model: nn.Module = model.to(self.device)
        if self.is_distributed:
            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            else:
                self.model = DistributedDataParallel(self.model)

        # Loss function
        self.loss_fn = LabelSmoothedNLLLoss(
            label_smoothing=config.label_smoothing,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Scheduler
        self.scheduler = NoamScheduler(
            self.optimizer,
            d_model=config.model.hidden_dim,
            warmup_steps=config.warmup_steps,
            factor=config.learning_rate_factor,
        )

        # Mixed precision
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # type: ignore[attr-defined]

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training state
        self.global_step = 0

        # Wandb
        self._wandb_run: Any = None

    @classmethod
    def from_config(cls, config: TrainingConfig) -> Trainer:
        """Build a Trainer from a TrainingConfig, constructing model and data loaders.

        Args:
            config: Full training configuration.

        Returns:
            Configured Trainer instance.
        """
        from teddympnn.data.collator import PaddingCollator
        from teddympnn.data.dataset import MixedDataLoader, PPIDataset

        _seed_everything(config.seed)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Build model
        is_ligand = config.model.model_type == "ligand_mpnn"
        model_cls = LigandMPNN if is_ligand else ProteinMPNN
        model_kwargs: dict[str, Any] = {
            "hidden_dim": config.model.hidden_dim,
            "num_encoder_layers": config.model.num_encoder_layers,
            "num_decoder_layers": config.model.num_decoder_layers,
            "num_neighbors": config.model.num_neighbors,
            "dropout": config.model.dropout_rate,
        }
        if is_ligand:
            model_kwargs["num_context_atoms"] = config.model.num_context_atoms
        model = model_cls(**model_kwargs)

        # Load pretrained weights
        load_model_weights(config.pretrained_weights, model)
        logger.info("Loaded pretrained weights from %s", config.pretrained_weights)

        # Build datasets
        collator = PaddingCollator()
        datasets = []
        weights = []
        for src in config.data_sources:
            ds = PPIDataset(
                manifest_path=src.path,
                max_residues=config.max_residues,
                min_interface_contacts=config.min_interface_contacts,
                include_ligand_atoms=is_ligand,
                atomize_partner_sidechains=is_ligand and config.atomize_partner_sidechains,
                sidechain_atomization_probability=config.sidechain_atomization_probability,
                sidechain_atomization_per_residue_probability=(
                    config.sidechain_atomization_per_residue_probability
                ),
            )
            datasets.append(ds)
            weights.append(src.weight)

        train_loader = MixedDataLoader(
            datasets=datasets,
            weights=weights,
            token_budget=config.token_budget,
            num_workers=config.num_workers,
            collate_fn=collator,
            weighted=True,
            rank=rank,
            world_size=world_size,
        )

        val_loader = None
        if config.validation_data_sources:
            val_datasets = []
            val_weights = []
            for src in config.validation_data_sources:
                ds = PPIDataset(
                    manifest_path=src.path,
                    max_residues=config.max_residues,
                    min_interface_contacts=config.min_interface_contacts,
                    include_ligand_atoms=is_ligand,
                    atomize_partner_sidechains=is_ligand and config.atomize_partner_sidechains,
                    sidechain_atomization_probability=1.0,
                    sidechain_atomization_per_residue_probability=1.0,
                )
                val_datasets.append(ds)
                val_weights.append(src.weight)
            val_loader = MixedDataLoader(
                datasets=val_datasets,
                weights=val_weights,
                token_budget=config.token_budget,
                num_workers=config.num_workers,
                collate_fn=collator,
                shuffle=False,
                weighted=False,
                rank=rank,
                world_size=world_size,
            )
        elif config.eval_every_n_steps and config.eval_every_n_steps < config.max_steps:
            logger.warning(
                "eval_every_n_steps=%d is set but no validation_data_sources are "
                "configured; validation will be skipped.",
                config.eval_every_n_steps,
            )

        return cls(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    def _init_wandb(self) -> None:
        """Initialize wandb logging if configured."""
        if self.config.wandb_project is None:
            return
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                config=self.config.model_dump(mode="json"),
            )
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging")

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to console and wandb."""
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        if self.rank == 0:
            logger.info("step %d: %s", step, ", ".join(parts))

        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def train_step(self, batch: dict[str, Any]) -> float:
        """Execute a single training step.

        Args:
            batch: Collated batch dict.

        Returns:
            Scalar loss value.
        """
        self.model.train()
        batch = self._move_batch(batch)

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):  # type: ignore[attr-defined]
            output = self.model(batch)
            loss = self.loss_fn(
                output["log_probs"],
                batch["S"],
                batch["designed_residue_mask"],
            )

        self.scaler.scale(loss).backward()

        if self.config.grad_clip_max_norm is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.grad_clip_max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return float(loss.item())

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation loop.

        Returns:
            Dict with validation metrics (loss, recovery, interface_recovery).
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_designed = 0
        total_iface_correct = 0
        total_iface = 0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch(batch)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):  # type: ignore[attr-defined]
                output = self.model(batch)
                loss = self.loss_fn(
                    output["log_probs"],
                    batch["S"],
                    batch["designed_residue_mask"],
                )

            total_loss += loss.item()
            n_batches += 1

            # Sequence recovery
            preds = output["log_probs"].argmax(dim=-1)  # (B, L)
            designed_mask = batch["designed_residue_mask"].bool()
            correct = (preds == batch["S"]) & designed_mask
            total_correct += correct.sum().item()
            total_designed += designed_mask.sum().item()

            B = preds.shape[0]
            for b in range(B):
                res_mask = batch["residue_mask"][b].bool()
                designed = designed_mask[b] & res_mask
                if not designed.any():
                    continue
                interface = identify_interface_residues(
                    batch["xyz_37"][b][res_mask],
                    batch["xyz_37_m"][b][res_mask],
                    batch["chain_labels"][b][res_mask],
                )
                full_interface = torch.zeros_like(res_mask)
                full_interface[res_mask] = interface
                designed_interface = designed & full_interface
                total_iface_correct += ((preds[b] == batch["S"][b]) & designed_interface).sum().item()
                total_iface += designed_interface.sum().item()

        if self.is_distributed:
            stats = torch.tensor(
                [
                    total_loss,
                    float(n_batches),
                    float(total_correct),
                    float(total_designed),
                    float(total_iface_correct),
                    float(total_iface),
                ],
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = float(stats[0].item())
            n_batches = int(stats[1].item())
            total_correct = int(stats[2].item())
            total_designed = int(stats[3].item())
            total_iface_correct = int(stats[4].item())
            total_iface = int(stats[5].item())

        metrics: dict[str, float] = {}
        if n_batches > 0:
            metrics["val_loss"] = total_loss / n_batches
        if total_designed > 0:
            metrics["val_recovery"] = total_correct / total_designed
        if total_iface > 0:
            metrics["val_interface_recovery"] = total_iface_correct / total_iface

        return metrics

    def save_checkpoint(self, step: int) -> Path:
        """Save a training checkpoint.

        Args:
            step: Current training step.

        Returns:
            Path to the saved checkpoint.
        """
        output_dir = Path(self.config.output_dir) / "checkpoints"
        if self.rank != 0:
            return output_dir / f"step_{step:07d}.pt"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"step_{step:07d}.pt"

        model_to_save = self.model
        if isinstance(model_to_save, nn.parallel.DistributedDataParallel):
            model_to_save = model_to_save.module

        save_checkpoint_bundle(
            path=path,
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            config=self.config.model_dump(mode="json"),
            model_family=self.config.model.model_type,
        )
        logger.info("Saved checkpoint at step %d to %s", step, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        model_to_load = self.model
        if isinstance(model_to_load, nn.parallel.DistributedDataParallel):
            model_to_load = model_to_load.module

        bundle = load_checkpoint_bundle(
            path=path,
            model=model_to_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.global_step = bundle.get("step", 0)
        logger.info("Resumed from step %d", self.global_step)

    def train(self) -> None:
        """Run the full training loop."""
        _seed_everything(self.config.seed)
        if self.rank == 0:
            self._init_wandb()

        if self.rank == 0:
            logger.info(
                "Starting training: max_steps=%d, device=%s, amp=%s, world_size=%d",
                self.config.max_steps,
                self.device,
                self.use_amp,
                self.world_size,
            )

        epoch = 0
        if hasattr(self.train_loader, "set_epoch"):
            self.train_loader.set_epoch(epoch)
        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            # Get next batch, cycling the data loader
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if hasattr(self.train_loader, "set_epoch"):
                    self.train_loader.set_epoch(epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            self.global_step += 1
            loss = self.train_step(batch)

            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self._log_metrics(
                    {"train_loss": loss, "lr": lr},
                    self.global_step,
                )

            # Validation
            if (
                self.val_loader is not None
                and self.global_step % self.config.eval_every_n_steps == 0
            ):
                val_metrics = self.validate()
                if val_metrics:
                    self._log_metrics(val_metrics, self.global_step)

            # Checkpointing
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(self.global_step)

        # Final checkpoint
        self.save_checkpoint(self.global_step)
        if self.rank == 0:
            logger.info("Training complete at step %d", self.global_step)

        if self._wandb_run is not None:
            import wandb

            wandb.finish()
        if self.is_distributed:
            dist.barrier()
