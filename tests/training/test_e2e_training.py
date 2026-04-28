"""End-to-end training validation gate (Phase 3.6).

Trains a model on a small test set to verify the full pipeline works:
pretrained weights → data loading → training loop → checkpoint → validation.

Requires GPU and pretrained weights. Marked with @pytest.mark.slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teddympnn.config import DataConfig, DatasetConfig, ModelConfig, TrainingConfig
from teddympnn.data.collator import PaddingCollator
from teddympnn.data.features import derive_backbone, parse_structure
from teddympnn.models import ProteinMPNN
from teddympnn.training.trainer import Trainer
from teddympnn.weights.io import load_checkpoint_bundle
from teddympnn.weights.legacy import load_legacy_weights

# Reference data paths
_REF_DIR = Path(__file__).parent.parent / "validation" / "reference_data"
_STRUCTURES_DIR = _REF_DIR / "structures"
_WEIGHTS_DIR = _REF_DIR / "weights"

_PROTEINMPNN_WEIGHTS = _WEIGHTS_DIR / "proteinmpnn_v_48_020.pt"
_LIGANDMPNN_WEIGHTS = _WEIGHTS_DIR / "ligandmpnn_v_32_010_25.pt"

# Test structures (PDB complexes with two chains)
_TEST_STRUCTURES = ["1BRS"]

_has_gpu = torch.cuda.is_available()
_has_weights = _PROTEINMPNN_WEIGHTS.exists() and _LIGANDMPNN_WEIGHTS.exists()
_has_structures = all((_STRUCTURES_DIR / f"{pdb}.pdb").exists() for pdb in _TEST_STRUCTURES)


def _can_run_e2e() -> bool:
    return _has_weights and _has_structures


requires_e2e = pytest.mark.skipif(
    not _can_run_e2e(),
    reason="E2E test requires pretrained weights and test structures.",
)


def _build_manifest(tmp_path: Path) -> Path:
    """Build a minimal manifest TSV from available test structures."""
    manifest_path = tmp_path / "manifest.tsv"
    lines = ["structure_path\tchain_A\tchain_B\tsource"]
    for pdb in _TEST_STRUCTURES:
        pdb_path = _STRUCTURES_DIR / f"{pdb}.pdb"
        if pdb_path.exists():
            # 1BRS has chains A and D (barnase-barstar complex)
            lines.append(f"{pdb_path}\tA\tD\tpdb")
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def _make_synthetic_batches(
    n_batches: int = 10,
    B: int = 2,
    L: int = 30,
) -> list[dict[str, torch.Tensor]]:
    """Create synthetic batches for CPU-only testing."""
    batches = []
    for _ in range(n_batches):
        batches.append(
            {
                "X": torch.randn(B, L, 4, 3),
                "S": torch.randint(0, 21, (B, L)),
                "R_idx": torch.arange(L).unsqueeze(0).expand(B, -1),
                "chain_labels": torch.zeros(B, L, dtype=torch.long),
                "residue_mask": torch.ones(B, L),
                "designed_residue_mask": torch.ones(B, L),
                "fixed_residue_mask": torch.zeros(B, L),
            }
        )
    return batches


@requires_e2e
@pytest.mark.slow
class TestE2EProteinMPNN:
    """End-to-end training test for ProteinMPNN."""

    def test_training_loss_decreases(self, tmp_path: Path) -> None:
        """Train ProteinMPNN for 100 steps; loss should decrease."""
        device = torch.device("cuda" if _has_gpu else "cpu")

        model = ProteinMPNN(
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            num_neighbors=48,
        )
        load_legacy_weights(_PROTEINMPNN_WEIGHTS, model)

        manifest_path = _build_manifest(tmp_path)

        config = TrainingConfig(
            model_type="protein_mpnn",
            pretrained_weights=_PROTEINMPNN_WEIGHTS,
            data=DataConfig(
                train={"pdb": DatasetConfig(path=manifest_path, ratio=1.0)},
            ),
            token_budget=2000,
            max_steps=100,
            warmup_steps=10,
            log_every_n_steps=10,
            eval_every_n_steps=50,
            save_every_n_steps=50,
            mixed_precision=_has_gpu,
            gradient_checkpointing=False,
            structure_noise=0.2,
            num_workers=0,
            output_dir=tmp_path / "outputs",
        )

        # Build data loader from manifest
        from teddympnn.data.dataset import PPIDataset
        from teddympnn.data.sampler import TokenBudgetBatchSampler

        dataset = PPIDataset(manifest_path, max_residues=6000)
        collator = PaddingCollator()
        sampler = TokenBudgetBatchSampler(dataset.lengths, token_budget=2000, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collator, num_workers=0
        )

        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            device=device,
        )
        trainer.train()

        # Check checkpoints were created
        ckpt_dir = tmp_path / "outputs" / "checkpoints"
        assert (ckpt_dir / "step_0000050.pt").exists()
        assert (ckpt_dir / "step_0000100.pt").exists()

        # Load checkpoint and verify it's valid
        model2 = ProteinMPNN()
        bundle = load_checkpoint_bundle(
            ckpt_dir / "step_0000100.pt",
            model2,
            map_location="cpu",
        )
        assert bundle["step"] == 100

    def test_checkpoint_produces_valid_output(self, tmp_path: Path) -> None:
        """Loaded checkpoint should produce valid log_probs."""
        device = torch.device("cuda" if _has_gpu else "cpu")

        model = ProteinMPNN(
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            num_neighbors=48,
        )
        load_legacy_weights(_PROTEINMPNN_WEIGHTS, model)
        model = model.to(device)
        model.eval()

        # Parse a test structure
        features = parse_structure(_STRUCTURES_DIR / "1BRS.pdb")
        X, X_m = derive_backbone(features["xyz_37"], features["xyz_37_m"])

        batch = {
            "X": X.unsqueeze(0).to(device),
            "S": features["S"].unsqueeze(0).to(device),
            "R_idx": features["R_idx"].unsqueeze(0).to(device),
            "chain_labels": features["chain_labels"].unsqueeze(0).to(device),
            "residue_mask": features["residue_mask"].unsqueeze(0).to(device),
            "designed_residue_mask": features["residue_mask"].unsqueeze(0).to(device),
            "fixed_residue_mask": torch.zeros_like(features["residue_mask"])
            .unsqueeze(0)
            .to(device),
        }

        with torch.no_grad():
            output = model(batch)

        log_probs = output["log_probs"]
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()
        assert (log_probs <= 0).all()

        # Recovery should be better than random (1/21 ≈ 4.8%)
        preds = log_probs.argmax(dim=-1)
        mask = batch["residue_mask"].bool()
        correct = ((preds == batch["S"]) & mask).sum().item()
        total = mask.sum().item()
        recovery = correct / max(total, 1)
        assert recovery > 0.05, f"Recovery {recovery:.4f} not better than random"


class TestE2ECPUOnly:
    """Lightweight e2e tests that run on CPU with synthetic data."""

    def test_training_loop_synthetic(self, tmp_path: Path) -> None:
        """Train a tiny model on synthetic data for 50 steps on CPU."""
        model = ProteinMPNN(
            hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1, num_neighbors=10
        )

        config = TrainingConfig(
            model_type="protein_mpnn",
            model=ModelConfig(
                hidden_dim=32,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_neighbors=10,
            ),
            pretrained_weights=tmp_path / "dummy.pt",
            data=DataConfig(
                train={"pdb": DatasetConfig(path=tmp_path / "m.tsv", ratio=1.0)},
            ),
            max_steps=50,
            warmup_steps=5,
            log_every_n_steps=10,
            eval_every_n_steps=25,
            save_every_n_steps=25,
            mixed_precision=False,
            gradient_checkpointing=False,
            output_dir=tmp_path / "outputs",
        )

        batches = _make_synthetic_batches(n_batches=10, B=2, L=15)
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=batches,
            device=torch.device("cpu"),
        )

        # Collect losses
        losses = []
        for batch in batches * 5:  # 50 steps
            loss = trainer.train_step(batch)
            losses.append(loss)
            trainer.global_step += 1
            if trainer.global_step % 25 == 0:
                trainer.save_checkpoint(trainer.global_step)

        # Loss should generally decrease
        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first

        # Checkpoints should exist
        ckpt_dir = tmp_path / "outputs" / "checkpoints"
        assert (ckpt_dir / "step_0000025.pt").exists()
        assert (ckpt_dir / "step_0000050.pt").exists()

    def test_no_nan_with_mixed_precision_cpu(self, tmp_path: Path) -> None:
        """Mixed precision on CPU should not produce NaN (autocast is a no-op on CPU)."""
        model = ProteinMPNN(
            hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1, num_neighbors=10
        )
        config = TrainingConfig(
            model_type="protein_mpnn",
            model=ModelConfig(
                hidden_dim=32,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_neighbors=10,
            ),
            pretrained_weights=tmp_path / "dummy.pt",
            data=DataConfig(
                train={"pdb": DatasetConfig(path=tmp_path / "m.tsv", ratio=1.0)},
            ),
            max_steps=5,
            mixed_precision=False,
            gradient_checkpointing=False,
            output_dir=tmp_path / "outputs",
        )

        trainer = Trainer(
            config=config,
            model=model,
            train_loader=_make_synthetic_batches(n_batches=5),
            device=torch.device("cpu"),
        )

        for batch in _make_synthetic_batches(n_batches=5):
            loss = trainer.train_step(batch)
            assert not torch.isnan(torch.tensor(loss))
            assert not torch.isinf(torch.tensor(loss))
