"""Trainer for teddyMPNN fine-tuning.

Supports single-GPU and multi-GPU (DDP) training with mixed precision,
gradient checkpointing, and Noam LR scheduling.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from teddympnn.models import LigandMPNN, ProteinMPNN
from teddympnn.training.loss import LabelSmoothedNLLLoss
from teddympnn.training.scheduler import NoamScheduler
from teddympnn.weights.io import load_checkpoint_bundle, save_checkpoint_bundle

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
        train_loader: DataLoader[dict[str, Any]],
        val_loader: DataLoader[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.model.use_gradient_checkpointing = True

        # Set structure noise from config
        self.model.augment_eps = config.structure_noise

        # Loss function
        self.loss_fn = LabelSmoothedNLLLoss(
            label_smoothing=config.label_smoothing,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0)

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
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

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

        # Build model
        is_ligand = config.model.model_type == "ligand_mpnn"
        model_cls = LigandMPNN if is_ligand else ProteinMPNN
        model = model_cls(
            hidden_dim=config.model.hidden_dim,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=config.model.num_decoder_layers,
            num_neighbors=config.model.num_neighbors,
            dropout=config.model.dropout_rate,
        )

        # Load pretrained weights
        from teddympnn.weights.legacy import load_legacy_weights

        load_legacy_weights(config.pretrained_weights, model)
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
            )
            datasets.append(ds)
            weights.append(src.weight)

        train_loader = MixedDataLoader(
            datasets=datasets,
            weights=weights,
            token_budget=config.token_budget,
            num_workers=config.num_workers,
            collate_fn=collator,
        )

        return cls(
            config=config,
            model=model,
            train_loader=train_loader,
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

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
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

        return loss.item()

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

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
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
        self._init_wandb()

        logger.info(
            "Starting training: max_steps=%d, device=%s, amp=%s",
            self.config.max_steps,
            self.device,
            self.use_amp,
        )

        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps:
            # Get next batch, cycling the data loader
            try:
                batch = next(data_iter)
            except StopIteration:
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
        logger.info("Training complete at step %d", self.global_step)

        if self._wandb_run is not None:
            import wandb

            wandb.finish()
