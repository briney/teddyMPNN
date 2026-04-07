"""Tests for teddymer data acquisition pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used in fixture type annotations

import pandas as pd
import pytest

from teddympnn.data.teddymer import (
    _parse_chopping,
    filter_teddymer_clusters,
)

# ---------------------------------------------------------------------------
# Chopping string parsing
# ---------------------------------------------------------------------------


class TestParseChopping:
    def test_simple_range(self):
        assert _parse_chopping("10-50") == [(10, 50)]

    def test_discontinuous(self):
        assert _parse_chopping("10-50,80-120") == [(10, 50), (80, 120)]

    def test_with_chain_prefix(self):
        assert _parse_chopping("A:10-50") == [(10, 50)]

    def test_with_chain_prefix_discontinuous(self):
        assert _parse_chopping("A:10-50,A:80-120") == [(10, 50), (80, 120)]

    def test_whitespace_handling(self):
        assert _parse_chopping(" 10 - 50 , 80 - 120 ") == [(10, 50), (80, 120)]


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


class TestFilterTeddymerClusters:
    @pytest.fixture
    def sample_metadata(self, tmp_path: Path) -> Path:
        """Create a sample metadata TSV for testing."""
        metadata_path = tmp_path / "nonsingletonrep_metadata.tsv"
        df = pd.DataFrame(
            {
                "cluster_id": [1, 2, 3, 4, 5],
                "InterfacePlddt": [80.0, 60.0, 75.0, 90.0, 50.0],
                "AvgIntPAE": [5.0, 15.0, 8.0, 3.0, 20.0],
                "InterfaceLength": [15, 12, 8, 20, 5],
                "uniprot_id": ["P00001", "P00002", "P00003", "P00004", "P00005"],
            }
        )
        df.to_csv(metadata_path, sep="\t", index=False)
        return tmp_path

    def test_filters_by_plddt(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_teddymer_clusters(sample_metadata, output)

        # Only entries with InterfacePlddt > 70
        assert all(df["interfaceplddt"] > 70.0)

    def test_filters_by_pae(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_teddymer_clusters(sample_metadata, output)

        # Only entries with AvgIntPAE < 10
        assert all(df["avgintpae"] < 10.0)

    def test_filters_by_interface_length(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_teddymer_clusters(sample_metadata, output)

        # Only entries with InterfaceLength > 10
        assert all(df["interfacelength"] > 10)

    def test_combined_filters(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_teddymer_clusters(sample_metadata, output)

        # Entries 1 and 4 should pass all filters:
        # 1: plddt=80>70, pae=5<10, ifl=15>10 ✓
        # 2: plddt=60<70 ✗
        # 3: plddt=75>70, pae=8<10, ifl=8<10 ✗
        # 4: plddt=90>70, pae=3<10, ifl=20>10 ✓
        # 5: plddt=50<70 ✗
        assert len(df) == 2

    def test_writes_output_file(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        filter_teddymer_clusters(sample_metadata, output)
        assert output.exists()
        reloaded = pd.read_csv(output, sep="\t")
        assert len(reloaded) == 2

    def test_custom_thresholds(self, sample_metadata: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_teddymer_clusters(
            sample_metadata,
            output,
            min_interface_plddt=50.0,
            max_interface_pae=20.0,
            min_interface_length=5,
        )
        # With relaxed filters, more entries should pass
        assert len(df) >= 2
