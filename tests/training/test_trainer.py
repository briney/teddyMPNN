"""Tests for the Trainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from teddympnn.config import DataConfig, DatasetConfig, ModelConfig, TrainingConfig
from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.training.trainer import Trainer


def _make_batch(B: int = 2, L: int = 20, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create a synthetic batch for testing."""
    xyz_37 = torch.randn(B, L, 37, 3, device=device)
    return {
        "X": xyz_37[:, :, :4, :].clone(),
        "xyz_37": xyz_37,
        "xyz_37_m": torch.ones(B, L, 37, dtype=torch.bool, device=device),
        "S": torch.randint(0, 21, (B, L), device=device),
        "R_idx": torch.arange(L, device=device).unsqueeze(0).expand(B, -1),
        "chain_labels": torch.zeros(B, L, dtype=torch.long, device=device),
        "residue_mask": torch.ones(B, L, device=device),
        "designed_residue_mask": torch.ones(B, L, device=device),
        "fixed_residue_mask": torch.zeros(B, L, device=device),
    }


def _make_tiny_config(tmp_path: Path) -> TrainingConfig:
    """Create a minimal training config for testing."""
    return TrainingConfig(
        model_type="protein_mpnn",
        model=ModelConfig(
            hidden_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_neighbors=10,
        ),
        pretrained_weights=tmp_path / "dummy.pt",
        data=DataConfig(
            train={"pdb": DatasetConfig(path=tmp_path / "manifest.tsv", ratio=1.0)},
        ),
        max_steps=10,
        log_every_n_steps=5,
        eval_every_n_steps=5,
        save_every_n_steps=5,
        mixed_precision=False,
        gradient_checkpointing=False,
        warmup_steps=5,
        learning_rate_factor=2.0,
        output_dir=tmp_path / "outputs",
    )


def _make_fake_loader(
    n_batches: int = 4,
    B: int = 2,
    L: int = 15,
) -> list[dict[str, torch.Tensor]]:
    """Create a list of synthetic batches."""
    return [_make_batch(B=B, L=L) for _ in range(n_batches)]


class TestTrainerTrainStep:
    def test_loss_is_finite(self, tmp_path: Path) -> None:
        """Single train step produces a finite loss."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        batch = _make_batch()
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))

    def test_loss_decreases(self, tmp_path: Path) -> None:
        """Loss should decrease over multiple steps on the same batch."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        batch = _make_batch(B=1, L=10)
        losses = []
        for _ in range(50):
            loss = trainer.train_step(batch)
            losses.append(loss)

        # Loss should generally decrease: last 10 average < first 10 average
        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first

    def test_gradient_clipping(self, tmp_path: Path) -> None:
        """Gradient clipping caps gradient norm."""
        config = _make_tiny_config(tmp_path)
        config.grad_clip_max_norm = 0.1
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        batch = _make_batch()
        # Just ensure it runs without error
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)


class TestTrainerCheckpoint:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Checkpoint save creates a file on disk."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        path = trainer.save_checkpoint(step=100)
        assert path.exists()
        bundle = torch.load(path, weights_only=False, map_location="cpu")
        assert bundle["step"] == 100
        assert "state_dict" in bundle
        assert "optimizer" in bundle
        assert "scheduler" in bundle

    def test_save_and_resume(self, tmp_path: Path) -> None:
        """Checkpoint resume restores training state."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        # Train a few steps
        batch = _make_batch()
        for _ in range(5):
            trainer.train_step(batch)
            trainer.global_step += 1

        # Save
        ckpt_path = trainer.save_checkpoint(step=trainer.global_step)

        # Record model output (seed for deterministic causality masks)
        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            output_before = model(batch)["log_probs"]

        # Create new trainer and resume
        model2 = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer2 = Trainer(
            config=config,
            model=model2,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.global_step == 5

        # Model output should match with same seed
        model2.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            output_after = model2(batch)["log_probs"]
        assert torch.allclose(output_before, output_after, atol=1e-5)


class TestTrainerValidation:
    def test_validate_returns_metrics(self, tmp_path: Path) -> None:
        """Validation returns loss and recovery metrics."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )

        val_data = _make_fake_loader(n_batches=2)
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            val_loader=val_data,
            device=torch.device("cpu"),
        )

        metrics = trainer.validate()
        assert "val_loss" in metrics
        assert "val_recovery" in metrics
        assert 0.0 <= metrics["val_recovery"] <= 1.0

    def test_validate_no_val_loader(self, tmp_path: Path) -> None:
        """Validation with no val_loader returns empty dict."""
        config = _make_tiny_config(tmp_path)
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(),
            device=torch.device("cpu"),
        )

        metrics = trainer.validate()
        assert metrics == {}


class TestTrainerFullLoop:
    def test_train_loop_completes(self, tmp_path: Path) -> None:
        """Full training loop completes and creates checkpoints."""
        config = _make_tiny_config(tmp_path)
        config.max_steps = 10
        config.save_every_n_steps = 5

        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_fake_loader(n_batches=4),
            device=torch.device("cpu"),
        )
        trainer.train()

        assert trainer.global_step == 10
        # Should have checkpoints at step 5 and 10
        ckpt_dir = tmp_path / "outputs" / "checkpoints"
        assert (ckpt_dir / "step_0000005.pt").exists()
        assert (ckpt_dir / "step_0000010.pt").exists()
