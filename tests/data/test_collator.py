"""Tests for padding collator."""

from __future__ import annotations

import pytest
import torch

from teddympnn.data.collator import PaddingCollator
from teddympnn.models.tokens import NUM_ATOMS_37


def _make_example(L: int) -> dict[str, torch.Tensor | int]:
    """Create a synthetic feature dict mimicking PPIDataset output."""
    return {
        "xyz_37": torch.randn(L, NUM_ATOMS_37, 3),
        "xyz_37_m": torch.ones(L, NUM_ATOMS_37, dtype=torch.bool),
        "X": torch.randn(L, 4, 3),
        "X_m": torch.ones(L, 4, dtype=torch.bool),
        "S": torch.randint(0, 21, (L,)),
        "R_idx": torch.arange(L),
        "chain_labels": torch.zeros(L, dtype=torch.long),
        "residue_mask": torch.ones(L, dtype=torch.bool),
        "designed_residue_mask": torch.ones(L, dtype=torch.bool),
        "fixed_residue_mask": torch.zeros(L, dtype=torch.bool),
        "num_residues": L,
    }


class TestPaddingCollator:
    def test_uniform_lengths(self):
        """Batch of same-length examples needs no padding."""
        collator = PaddingCollator()
        batch = [_make_example(50), _make_example(50)]
        result = collator(batch)

        assert result["S"].shape == (2, 50)
        assert result["xyz_37"].shape == (2, 50, NUM_ATOMS_37, 3)

    def test_variable_lengths(self):
        """Shorter examples should be padded to the longest."""
        collator = PaddingCollator()
        batch = [_make_example(30), _make_example(50), _make_example(40)]
        result = collator(batch)

        L_max = 50
        assert result["S"].shape == (3, L_max)
        assert result["xyz_37"].shape == (3, L_max, NUM_ATOMS_37, 3)
        assert result["X"].shape == (3, L_max, 4, 3)

    def test_padding_values_residue_mask(self):
        """Padded positions should have residue_mask=False."""
        collator = PaddingCollator()
        batch = [_make_example(30), _make_example(50)]
        result = collator(batch)

        # First example: positions 30-49 should be False
        assert result["residue_mask"][0, :30].all()
        assert not result["residue_mask"][0, 30:].any()

        # Second example: all positions valid
        assert result["residue_mask"][1, :50].all()

    def test_padding_values_token(self):
        """Padded S positions should be UNK (20)."""
        collator = PaddingCollator()
        batch = [_make_example(20), _make_example(40)]
        result = collator(batch)

        assert (result["S"][0, 20:] == 20).all()

    def test_padding_values_chain_labels(self):
        """Padded chain_labels should be -1."""
        collator = PaddingCollator()
        batch = [_make_example(20), _make_example(40)]
        result = collator(batch)

        assert (result["chain_labels"][0, 20:] == -1).all()

    def test_padding_values_coordinates(self):
        """Padded xyz_37 positions should be 0.0."""
        collator = PaddingCollator()
        batch = [_make_example(20), _make_example(40)]
        result = collator(batch)

        assert (result["xyz_37"][0, 20:] == 0.0).all()

    def test_metadata_collected_as_list(self):
        """Non-tensor values should be collected into lists."""
        collator = PaddingCollator()
        batch = [_make_example(30), _make_example(40)]
        result = collator(batch)

        assert isinstance(result["num_residues"], list)
        assert result["num_residues"] == [30, 40]

    def test_dtypes_preserved(self):
        """Output dtypes should match input dtypes."""
        collator = PaddingCollator()
        batch = [_make_example(30), _make_example(50)]
        result = collator(batch)

        assert result["xyz_37"].dtype == torch.float32
        assert result["S"].dtype == torch.int64
        assert result["residue_mask"].dtype == torch.bool
        assert result["chain_labels"].dtype == torch.int64

    def test_designed_mask_padded_false(self):
        """Padded designed_residue_mask should be False."""
        collator = PaddingCollator()
        batch = [_make_example(20), _make_example(40)]
        result = collator(batch)

        assert not result["designed_residue_mask"][0, 20:].any()

    def test_empty_batch_raises(self):
        collator = PaddingCollator()
        with pytest.raises(ValueError, match="empty"):
            collator([])
