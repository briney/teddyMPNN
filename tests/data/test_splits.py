"""Tests for train/val manifest splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from teddympnn.data.splits import (
    MANIFEST_COLUMNS,
    _hash_split,
    prepare_manifests,
    split_nvidia_manifest,
    split_pdb_manifest,
    split_teddymer_manifest,
)


@pytest.fixture()
def teddymer_manifest(tmp_path: Path) -> Path:
    """Create a synthetic teddymer manifest with cluster_rep column."""
    rows = []
    for cluster_id in range(20):
        for member in range(5):
            rows.append(
                {
                    "cluster_rep": f"cluster_{cluster_id}",
                    "uniprot_id": f"Q{cluster_id:04d}{member}",
                    "structure_path": f"/data/teddymer/{cluster_id}_{member}.pdb",
                    "chain_A": "A",
                    "chain_B": "B",
                    "interfaceplddt": 85.0,
                    "avgintpae": 5.0,
                    "interfacelength": 20,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "teddymer_manifest.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture()
def nvidia_manifest(tmp_path: Path) -> Path:
    """Create a synthetic NVIDIA complexes manifest."""
    rows = []
    for i in range(50):
        rows.append(
            {
                "model_id": f"AF-{i:04d}",
                "structure_path": f"/data/nvidia/{i}.cif",
                "chain_A": "A",
                "chain_B": "B",
                "ipSAEmin": 0.8,
                "pLDDTavg": 80.0,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "nvidia_manifest.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture()
def pdb_manifest(tmp_path: Path) -> Path:
    """Create a synthetic PDB complexes manifest."""
    rows = []
    for i in range(40):
        pdb_id = f"{i + 1000:04d}"
        rows.append(
            {
                "pdb_id": pdb_id,
                "structure_path": f"/data/pdb/{pdb_id}.cif",
                "chain_A": "A",
                "chain_B": "B",
                "num_chains": 2,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "pdb_manifest.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


class TestHashSplit:
    def test_deterministic(self) -> None:
        """Same key and seed always produce the same result."""
        for _ in range(10):
            assert _hash_split("test_key", 0.05, seed=42) == _hash_split("test_key", 0.05, seed=42)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different assignments for at least some keys."""
        results_seed1 = [_hash_split(f"key_{i}", 0.5, seed=1) for i in range(100)]
        results_seed2 = [_hash_split(f"key_{i}", 0.5, seed=2) for i in range(100)]
        assert results_seed1 != results_seed2

    def test_approximate_fraction(self) -> None:
        """Fraction of keys assigned to val is approximately correct."""
        val_fraction = 0.20
        n = 10_000
        n_val = sum(_hash_split(f"key_{i}", val_fraction, seed=42) for i in range(n))
        actual_fraction = n_val / n
        assert abs(actual_fraction - val_fraction) < 0.02


class TestSplitTeddymer:
    def test_split_by_cluster(self, teddymer_manifest: Path) -> None:
        """All members of a cluster go to the same split."""
        train_df, val_df = split_teddymer_manifest(teddymer_manifest)

        # No cluster appears in both
        train_clusters = set(train_df["cluster_rep"].unique())
        val_clusters = set(val_df["cluster_rep"].unique())
        assert train_clusters.isdisjoint(val_clusters)

        # All rows accounted for
        assert len(train_df) + len(val_df) == 100  # 20 clusters * 5 members

    def test_reproducible(self, teddymer_manifest: Path) -> None:
        """Same seed produces identical splits."""
        t1, v1 = split_teddymer_manifest(teddymer_manifest, seed=42)
        t2, v2 = split_teddymer_manifest(teddymer_manifest, seed=42)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))
        pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))

    def test_val_fraction(self, teddymer_manifest: Path) -> None:
        """Validation fraction is approximately correct."""
        train_df, val_df = split_teddymer_manifest(teddymer_manifest, val_fraction=0.20)
        val_clusters = len(val_df["cluster_rep"].unique())
        assert 2 <= val_clusters <= 6  # ~20% of 20 clusters


class TestSplitNvidia:
    def test_split_by_complex(self, nvidia_manifest: Path) -> None:
        """Each complex goes to exactly one split."""
        train_df, val_df = split_nvidia_manifest(nvidia_manifest)
        train_ids = set(train_df["model_id"].unique())
        val_ids = set(val_df["model_id"].unique())
        assert train_ids.isdisjoint(val_ids)
        assert len(train_df) + len(val_df) == 50


class TestSplitPdb:
    def test_split_by_structure(self, pdb_manifest: Path) -> None:
        """Each structure goes to exactly one split."""
        train_df, val_df = split_pdb_manifest(pdb_manifest)
        train_ids = set(train_df["pdb_id"].unique())
        val_ids = set(val_df["pdb_id"].unique())
        assert train_ids.isdisjoint(val_ids)
        assert len(train_df) + len(val_df) == 40


class TestPrepareManifests:
    def test_single_source(self, tmp_path: Path, pdb_manifest: Path) -> None:
        """Works with a single data source."""
        train_path, val_path = prepare_manifests(tmp_path / "manifests", pdb_manifest=pdb_manifest)
        assert train_path.exists()
        assert val_path.exists()

        train_df = pd.read_csv(train_path, sep="\t")
        val_df = pd.read_csv(val_path, sep="\t")
        assert tuple(train_df.columns) == MANIFEST_COLUMNS
        assert (train_df["source"] == "pdb").all()
        assert len(train_df) + len(val_df) == 40
        assert (tmp_path / "manifests" / "train_pdb.tsv").exists()
        assert (tmp_path / "manifests" / "val_pdb.tsv").exists()

    def test_multiple_sources(
        self,
        tmp_path: Path,
        teddymer_manifest: Path,
        nvidia_manifest: Path,
        pdb_manifest: Path,
    ) -> None:
        """Combines all three sources into unified manifests."""
        train_path, val_path = prepare_manifests(
            tmp_path / "manifests",
            teddymer_manifest=teddymer_manifest,
            nvidia_manifest=nvidia_manifest,
            pdb_manifest=pdb_manifest,
        )

        train_df = pd.read_csv(train_path, sep="\t")
        val_df = pd.read_csv(val_path, sep="\t")

        # All three sources present in train
        assert set(train_df["source"].unique()) == {"teddymer", "nvidia", "pdb"}

        # Total row count preserved
        assert len(train_df) + len(val_df) == 100 + 50 + 40

        # Split stats file written
        stats_path = tmp_path / "manifests" / "split_stats.tsv"
        assert stats_path.exists()
        for source in ["teddymer", "nvidia", "pdb"]:
            assert (tmp_path / "manifests" / f"train_{source}.tsv").exists()
            assert (tmp_path / "manifests" / f"val_{source}.tsv").exists()

    def test_no_sources_raises(self, tmp_path: Path) -> None:
        """Raises ValueError when no sources are provided."""
        with pytest.raises(ValueError, match="At least one"):
            prepare_manifests(tmp_path / "manifests")

    def test_reproducible(
        self, tmp_path: Path, teddymer_manifest: Path, pdb_manifest: Path
    ) -> None:
        """Same seed produces identical unified manifests."""
        t1, v1 = prepare_manifests(
            tmp_path / "m1",
            teddymer_manifest=teddymer_manifest,
            pdb_manifest=pdb_manifest,
            seed=42,
        )
        t2, v2 = prepare_manifests(
            tmp_path / "m2",
            teddymer_manifest=teddymer_manifest,
            pdb_manifest=pdb_manifest,
            seed=42,
        )
        pd.testing.assert_frame_equal(pd.read_csv(t1, sep="\t"), pd.read_csv(t2, sep="\t"))
        pd.testing.assert_frame_equal(pd.read_csv(v1, sep="\t"), pd.read_csv(v2, sep="\t"))
