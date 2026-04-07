"""Tests for the unified PPI dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from teddympnn.data.dataset import PPIDataset
from teddympnn.models.tokens import NUM_ATOMS_37

# Reference structures
REFERENCE_DIR = Path(__file__).parent.parent / "validation" / "reference_data" / "structures"
PDB_1BRS = REFERENCE_DIR / "1BRS.pdb"


@pytest.fixture
def requires_reference_structures():
    if not PDB_1BRS.exists():
        pytest.skip("Reference structures not available")


@pytest.fixture
def manifest_with_1brs(tmp_path: Path, requires_reference_structures) -> Path:
    """Create a manifest TSV pointing to 1BRS."""
    manifest_path = tmp_path / "manifest.tsv"
    df = pd.DataFrame(
        {
            "structure_path": [str(PDB_1BRS)],
            "chain_A": ["A"],
            "chain_B": ["D"],
            "source": ["pdb"],
        }
    )
    df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


class TestPPIDataset:
    def test_partner_design_expansion(self, manifest_with_1brs: Path):
        """Each structure should produce two views."""
        ds = PPIDataset(manifest_with_1brs)
        # 1 structure → 2 views (design A|B and design B|A)
        assert len(ds) == 2

    def test_returns_expected_keys(self, manifest_with_1brs: Path):
        ds = PPIDataset(manifest_with_1brs)
        item = ds[0]

        expected_keys = {
            "xyz_37",
            "xyz_37_m",
            "X",
            "X_m",
            "S",
            "R_idx",
            "chain_labels",
            "residue_mask",
            "designed_residue_mask",
            "fixed_residue_mask",
            "Y",
            "Y_m",
            "Y_t",
            "num_residues",
        }
        assert expected_keys <= set(item.keys())

    def test_tensor_shapes(self, manifest_with_1brs: Path):
        ds = PPIDataset(manifest_with_1brs)
        item = ds[0]
        L = item["num_residues"]

        assert item["xyz_37"].shape == (L, NUM_ATOMS_37, 3)
        assert item["xyz_37_m"].shape == (L, NUM_ATOMS_37)
        assert item["X"].shape == (L, 4, 3)
        assert item["X_m"].shape == (L, 4)
        assert item["S"].shape == (L,)
        assert item["R_idx"].shape == (L,)
        assert item["chain_labels"].shape == (L,)
        assert item["residue_mask"].shape == (L,)
        assert item["designed_residue_mask"].shape == (L,)
        assert item["fixed_residue_mask"].shape == (L,)

    def test_partner_masks_complementary(self, manifest_with_1brs: Path):
        """Designed and fixed masks should not overlap."""
        ds = PPIDataset(manifest_with_1brs)
        item = ds[0]

        designed = item["designed_residue_mask"]
        fixed = item["fixed_residue_mask"]

        # No overlap
        assert not (designed & fixed).any()

    def test_views_have_opposite_masks(self, manifest_with_1brs: Path):
        """View 0 and view 1 should have swapped designed/fixed masks."""
        ds = PPIDataset(manifest_with_1brs)
        view0 = ds[0]
        view1 = ds[1]

        # Design targets should be swapped
        torch.testing.assert_close(
            view0["designed_residue_mask"],
            view1["fixed_residue_mask"],
        )
        torch.testing.assert_close(
            view0["fixed_residue_mask"],
            view1["designed_residue_mask"],
        )

    def test_designed_mask_covers_one_chain(self, manifest_with_1brs: Path):
        """Each view should design exactly one chain."""
        ds = PPIDataset(manifest_with_1brs)
        item = ds[0]

        designed = item["designed_residue_mask"]
        chain_labels = item["chain_labels"]

        designed_chains = chain_labels[designed].unique()
        assert len(designed_chains) == 1, "Should design exactly one chain"

    def test_backbone_derived_correctly(self, manifest_with_1brs: Path):
        """X should match the first 4 atoms of xyz_37."""
        ds = PPIDataset(manifest_with_1brs)
        item = ds[0]

        torch.testing.assert_close(item["X"], item["xyz_37"][:, :4, :])

    def test_max_residues_filter(self, manifest_with_1brs: Path):
        """Structures exceeding max_residues should be skipped."""
        ds = PPIDataset(manifest_with_1brs, max_residues=10)
        assert len(ds) == 0, "1BRS should exceed 10 residues"

    def test_cache_produces_same_output(self, manifest_with_1brs: Path, tmp_path: Path):
        """Cached loading should produce identical features."""
        cache_dir = tmp_path / "cache"

        ds1 = PPIDataset(manifest_with_1brs, cache_dir=cache_dir)
        item1 = ds1[0]

        # Second load should use cache
        ds2 = PPIDataset(manifest_with_1brs, cache_dir=cache_dir)
        item2 = ds2[0]

        for key in ["S", "R_idx", "chain_labels"]:
            torch.testing.assert_close(item1[key], item2[key])

    def test_empty_ligand_context(self, manifest_with_1brs: Path):
        """Without include_ligand_atoms, Y should be empty."""
        ds = PPIDataset(manifest_with_1brs, include_ligand_atoms=False)
        item = ds[0]

        assert item["Y"].shape[0] == 0
        assert item["Y_m"].shape[0] == 0
        assert item["Y_t"].shape[0] == 0

    def test_lengths_property(self, manifest_with_1brs: Path):
        """lengths should match the number of views."""
        ds = PPIDataset(manifest_with_1brs)
        assert len(ds.lengths) == len(ds)
        assert all(n > 0 for n in ds.lengths)
