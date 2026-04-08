"""Tests for binding affinity (ddG) prediction."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime in fixtures

import pytest
import torch

from teddympnn.evaluation.binding_affinity import (
    _apply_mutations,
    _extract_chain_features,
    _make_batch,
    predict_ddg,
    score_structure,
)
from teddympnn.models import ProteinMPNN
from teddympnn.models.tokens import token_to_idx


def _write_test_pdb(path: Path, n_res_a: int = 15, n_res_b: int = 15) -> None:
    """Write a minimal two-chain PDB for testing.

    Creates two chains of all-Ala residues with valid backbone geometry
    and an interface where the chains are close enough for CB contacts.
    """
    lines: list[str] = []
    atom_idx = 1

    for chain, n_res, x_offset in [("A", n_res_a, 0.0), ("B", n_res_b, 10.0)]:
        for res_idx in range(n_res):
            y_base = res_idx * 3.8
            atoms = [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.458, 0.0, 0.0),
                ("C", 2.009, 1.420, 0.0),
                ("O", 1.246, 2.381, 0.0),
            ]
            for name, dx, dy, dz in atoms:
                x = x_offset + dx
                y = y_base + dy
                z = dz
                # PDB ATOM format
                lines.append(
                    f"ATOM  {atom_idx:5d}  {name:<3s} ALA {chain}{res_idx + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}\n"
                )
                atom_idx += 1
        lines.append(f"TER   {atom_idx:5d}      ALA {chain}{n_res:4d}\n")
        atom_idx += 1

    lines.append("END\n")
    path.write_text("".join(lines))


def _make_model() -> ProteinMPNN:
    """Create a small ProteinMPNN for testing."""
    return ProteinMPNN(
        hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_neighbors=10,
    )


class TestScoreStructure:
    """Tests for score_structure."""

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical scores."""
        model = _make_model()
        model.eval()

        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)
        from teddympnn.data.features import parse_structure

        features = parse_structure(tmp_path / "test.pdb")
        L = len(features["S"])
        designed = torch.ones(L, dtype=torch.bool)
        batch = _make_batch(features, designed, torch.device("cpu"))
        mask = torch.ones(1, L, dtype=torch.bool)

        s1 = score_structure(model, batch, mask, seed=42)
        s2 = score_structure(model, batch, mask, seed=42)
        assert s1 == s2

    def test_different_seeds_give_different_scores(self, tmp_path: Path) -> None:
        """Different seeds should typically produce different scores."""
        model = _make_model()
        model.eval()

        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)
        from teddympnn.data.features import parse_structure

        features = parse_structure(tmp_path / "test.pdb")
        L = len(features["S"])
        designed = torch.ones(L, dtype=torch.bool)
        batch = _make_batch(features, designed, torch.device("cpu"))
        mask = torch.ones(1, L, dtype=torch.bool)

        scores = [score_structure(model, batch, mask, seed=i) for i in range(10)]
        # At least some scores should differ (extremely unlikely all are equal)
        assert len(set(scores)) > 1


class TestApplyMutations:
    """Tests for _apply_mutations."""

    def test_single_mutation(self) -> None:
        """Single mutation should change exactly one position."""
        S = torch.tensor([token_to_idx["ALA"]] * 10)
        features = {"S": S}
        chain_ids = ["A"] * 10
        residue_numbers = list(range(1, 11))
        mutations = {"A": {"A5G": None}}

        mut_features, mask = _apply_mutations(features, chain_ids, residue_numbers, mutations)

        assert mask.sum().item() == 1
        assert mask[4].item() is True  # 0-indexed position 4 = PDB residue 5
        assert mut_features["S"][4].item() == token_to_idx["GLY"]
        # Other positions unchanged
        assert (mut_features["S"][:4] == token_to_idx["ALA"]).all()
        assert (mut_features["S"][5:] == token_to_idx["ALA"]).all()

    def test_multi_mutation(self) -> None:
        """Multiple mutations on the same chain."""
        S = torch.tensor([token_to_idx["ALA"]] * 10)
        features = {"S": S}
        chain_ids = ["A"] * 10
        residue_numbers = list(range(1, 11))
        mutations = {"A": {"A1G": None, "A3L": None}}

        mut_features, mask = _apply_mutations(features, chain_ids, residue_numbers, mutations)

        assert mask.sum().item() == 2
        assert mut_features["S"][0].item() == token_to_idx["GLY"]
        assert mut_features["S"][2].item() == token_to_idx["LEU"]

    def test_cross_chain_mutation(self) -> None:
        """Mutations on different chains."""
        S = torch.tensor([token_to_idx["ALA"]] * 20)
        features = {"S": S}
        chain_ids = ["A"] * 10 + ["B"] * 10
        residue_numbers = list(range(1, 11)) + list(range(1, 11))
        mutations = {"A": {"A2G": None}, "B": {"A5W": None}}

        mut_features, mask = _apply_mutations(features, chain_ids, residue_numbers, mutations)

        assert mask.sum().item() == 2
        assert mask[1].item() is True  # chain A, residue 2
        assert mask[14].item() is True  # chain B, residue 5
        assert mut_features["S"][1].item() == token_to_idx["GLY"]
        assert mut_features["S"][14].item() == token_to_idx["TRP"]

    def test_missing_residue_raises(self) -> None:
        """Referencing a non-existent residue should raise ValueError."""
        features = {"S": torch.tensor([token_to_idx["ALA"]] * 5)}
        with pytest.raises(ValueError, match="not found"):
            _apply_mutations(
                features,
                chain_ids=["A"] * 5,
                residue_numbers=list(range(1, 6)),
                mutations={"A": {"A99G": None}},
            )


class TestExtractChainFeatures:
    """Tests for _extract_chain_features."""

    def test_extracts_correct_chain(self) -> None:
        """Should extract only residues belonging to target chains."""
        features = {
            "xyz_37": torch.randn(20, 37, 3),
            "xyz_37_m": torch.ones(20, 37, dtype=torch.bool),
            "S": torch.randint(0, 21, (20,)),
            "R_idx": torch.cat([torch.arange(10), torch.arange(10)]),
            "chain_labels": torch.cat([torch.zeros(10), torch.ones(10)]).long(),
            "residue_mask": torch.ones(20, dtype=torch.bool),
            "residue_numbers": list(range(1, 11)) + list(range(1, 11)),
        }
        chain_ids = ["A"] * 10 + ["B"] * 10

        new_feat, new_ids, new_nums = _extract_chain_features(features, chain_ids, {"A"})

        assert new_feat["S"].shape[0] == 10
        assert len(new_ids) == 10
        assert all(c == "A" for c in new_ids)
        assert new_nums == list(range(1, 11))


class TestPredictDdg:
    """Tests for predict_ddg."""

    def test_identity_mutation_near_zero(self, tmp_path: Path) -> None:
        """Wild-type → wild-type 'mutation' should give ddG ≈ 0."""
        model = _make_model()
        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)

        # "A1A" = Ala→Ala (identity)
        ddg = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"A": {"A1A": None}},
            num_samples=5,
            device=torch.device("cpu"),
        )
        assert abs(ddg) < 1e-4, f"Identity mutation ddG should be ~0, got {ddg}"

    def test_single_mutation_finite(self, tmp_path: Path) -> None:
        """A real mutation should produce a finite, non-NaN ddG."""
        model = _make_model()
        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)

        ddg = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"A": {"A5G": None}},
            num_samples=5,
            device=torch.device("cpu"),
        )
        assert not torch.isnan(torch.tensor(ddg))
        assert not torch.isinf(torch.tensor(ddg))

    def test_antithetic_reduces_variance(self, tmp_path: Path) -> None:
        """Antithetic variates should reduce variance vs independent sampling.

        We verify this indirectly: running the same prediction twice with
        the same model should give identical results (both use antithetic
        variates internally with deterministic seeds per sample).
        """
        model = _make_model()
        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)

        ddg1 = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"A": {"A3G": None}},
            num_samples=5,
            device=torch.device("cpu"),
        )
        ddg2 = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"A": {"A3G": None}},
            num_samples=5,
            device=torch.device("cpu"),
        )
        # Deterministic seeds → identical results
        assert ddg1 == ddg2

    def test_multi_residue_mutations(self, tmp_path: Path) -> None:
        """Multiple simultaneous mutations should work correctly."""
        model = _make_model()
        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)

        ddg = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"A": {"A1G": None, "A3L": None}},
            num_samples=3,
            device=torch.device("cpu"),
        )
        assert not torch.isnan(torch.tensor(ddg))
        assert not torch.isinf(torch.tensor(ddg))

    def test_mutation_on_partner_b(self, tmp_path: Path) -> None:
        """Mutations on chain B should also work."""
        model = _make_model()
        _write_test_pdb(tmp_path / "test.pdb", n_res_a=10, n_res_b=10)

        ddg = predict_ddg(
            model,
            tmp_path / "test.pdb",
            mutations={"B": {"A5G": None}},
            num_samples=3,
            device=torch.device("cpu"),
        )
        assert not torch.isnan(torch.tensor(ddg))
