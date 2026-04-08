"""Tests for interface sequence recovery evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from teddympnn.evaluation.sequence_recovery import RecoveryResults, compute_recovery
from teddympnn.models import ProteinMPNN


def _make_batch(
    B: int = 2,
    L: int = 30,
    n_chains: int = 2,
    interface_distance: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Create a synthetic batch with two chains and an interface.

    Chain A is residues 0..L//2, chain B is L//2..L.
    Interface residues are positioned close together; non-interface far apart.
    """
    half = L // 2
    chain_labels = torch.zeros(B, L, dtype=torch.long)
    chain_labels[:, half:] = 1

    # Build backbone coords: chain A near origin, chain B offset
    # Interface residues (last of A, first of B) are close
    X = torch.zeros(B, L, 4, 3)
    for b in range(B):
        for i in range(L):
            if i < half:
                # Chain A along x-axis
                X[b, i, :, 0] = i * 3.8  # N, CA, C, O roughly along x
            else:
                # Chain B: first few close to end of chain A (interface)
                if i < half + 3:
                    X[b, i, :, 0] = (half - 1) * 3.8 + interface_distance
                    X[b, i, :, 1] = (i - half) * 3.8
                else:
                    X[b, i, :, 0] = (i - half) * 3.8 + 100.0  # far away
            # Slightly offset backbone atoms
            X[b, i, 0, 1] = 0.0  # N
            X[b, i, 1, 1] = 1.5  # CA
            X[b, i, 2, 1] = 2.3  # C
            X[b, i, 3, 1] = 3.5  # O

    # Build xyz_37 from X (put backbone in first 4 slots, CB at slot 4)
    xyz_37 = torch.zeros(B, L, 37, 3)
    xyz_37[:, :, :4, :] = X
    # Virtual CB ≈ CA offset (crude approximation for testing)
    xyz_37[:, :, 4, :] = X[:, :, 1, :] + torch.tensor([0.0, 0.0, 1.5])

    xyz_37_m = torch.zeros(B, L, 37, dtype=torch.bool)
    xyz_37_m[:, :, :5] = True  # N, CA, C, O, CB resolved

    # Design chain B, condition on chain A
    designed_mask = torch.zeros(B, L, dtype=torch.bool)
    designed_mask[:, half:] = True

    return {
        "X": X,
        "xyz_37": xyz_37,
        "xyz_37_m": xyz_37_m,
        "S": torch.randint(0, 21, (B, L)),
        "R_idx": torch.arange(L).unsqueeze(0).expand(B, -1),
        "chain_labels": chain_labels,
        "residue_mask": torch.ones(B, L, dtype=torch.bool),
        "designed_residue_mask": designed_mask,
        "fixed_residue_mask": ~designed_mask,
    }


class _PerfectModel(ProteinMPNN):
    """Model that always predicts the ground truth sequence."""

    def forward(self, input_features, **kwargs):
        S = input_features["S"]
        B, L = S.shape
        log_probs = torch.full((B, L, self.vocab_size), -10.0)
        log_probs.scatter_(2, S.unsqueeze(-1), 0.0)
        return {"log_probs": log_probs}


class _RandomModel(ProteinMPNN):
    """Model that outputs uniform distribution over all amino acids."""

    def forward(self, input_features, **kwargs):
        S = input_features["S"]
        B, L = S.shape
        import math

        log_probs = torch.full((B, L, self.vocab_size), math.log(1.0 / self.vocab_size))
        return {"log_probs": log_probs}


def _make_loader(n_batches: int = 3, **kwargs) -> DataLoader:
    """Create a DataLoader of synthetic batches."""
    batches = [_make_batch(**kwargs) for _ in range(n_batches)]
    return DataLoader(batches, batch_size=None)


class TestComputeRecovery:
    """Tests for compute_recovery."""

    def test_perfect_model_100_percent(self) -> None:
        """A model returning ground truth should achieve 100% recovery."""
        model = _PerfectModel(hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1)
        loader = _make_loader(n_batches=3, B=2, L=20)
        results = compute_recovery(model, loader, device=torch.device("cpu"))

        assert isinstance(results, RecoveryResults)
        assert results.overall_recovery == 1.0
        assert results.n_structures > 0
        assert results.n_designed_residues > 0

    def test_random_model_near_chance(self) -> None:
        """A uniform model should achieve ~1/21 ≈ 4.8% recovery."""
        model = _RandomModel(hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1)
        # Use many batches for a stable estimate
        loader = _make_loader(n_batches=20, B=4, L=40)
        results = compute_recovery(model, loader, device=torch.device("cpu"))

        # With 20*4*20=1600 designed residues, random should be ~4.8%
        # Allow generous tolerance: 1% to 12%
        assert 0.01 < results.overall_recovery < 0.12

    def test_interface_mask_selects_subset(self) -> None:
        """Interface recovery should use fewer residues than overall."""
        model = _PerfectModel(hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1)
        # Use close interface distance so some but not all residues are interface
        loader = _make_loader(n_batches=3, B=2, L=30, interface_distance=5.0)
        results = compute_recovery(model, loader, interface_cutoff=8.0, device=torch.device("cpu"))

        # Interface residues should be a subset of designed residues
        assert results.n_interface_residues <= results.n_designed_residues
        # With our geometry, there should be some interface residues
        assert results.n_interface_residues > 0
        # Perfect model should still get 100% at interface
        assert results.interface_recovery == 1.0

    def test_per_structure_recovery_is_macroaveraged(self) -> None:
        """Per-structure recovery should be the mean of per-structure values."""
        model = _PerfectModel(hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1)
        loader = _make_loader(n_batches=2, B=3, L=20)
        results = compute_recovery(model, loader, device=torch.device("cpu"))

        # For perfect model, both micro and macro should be 1.0
        assert results.per_structure_recovery == 1.0

    def test_empty_loader(self) -> None:
        """Empty loader should return zero metrics without crashing."""
        model = _PerfectModel(hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1)
        loader = DataLoader([], batch_size=None)
        results = compute_recovery(model, loader, device=torch.device("cpu"))

        assert results.n_structures == 0
        assert results.n_designed_residues == 0
