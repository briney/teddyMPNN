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
        """Create a sample metadata TSV with start/end domain columns."""
        metadata_path = tmp_path / "nonsingletonrep_metadata.tsv"
        df = pd.DataFrame(
            {
                "cluster_id": [1, 2, 3, 4, 5],
                "InterfacePlddt": [80.0, 60.0, 75.0, 90.0, 50.0],
                "AvgIntPAE": [5.0, 15.0, 8.0, 3.0, 20.0],
                "InterfaceLength": [15, 12, 8, 20, 5],
                "uniprot_id": ["P00001", "P00002", "P00003", "P00004", "P00005"],
                "domain1_start": [10, 20, 30, 40, 50],
                "domain1_end": [80, 90, 100, 110, 120],
                "domain2_start": [150, 160, 170, 180, 190],
                "domain2_end": [220, 230, 240, 250, 260],
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


# ---------------------------------------------------------------------------
# Chopping column derivation
# ---------------------------------------------------------------------------


class TestChoppingDerivation:
    """Verify ``filter_teddymer_clusters`` produces the chopping columns
    required by ``chop_and_assemble_dimers`` from each supported input schema.
    """

    def _write_metadata(self, dir_: Path, df: pd.DataFrame) -> Path:
        path = dir_ / "nonsingletonrep_metadata.tsv"
        df.to_csv(path, sep="\t", index=False)
        return dir_

    def test_passthrough_when_chopping_present(self, tmp_path: Path):
        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "uniprot_id": ["P00001"],
                    "domain1_chopping": ["10-50"],
                    "domain2_chopping": ["100-200,250-300"],
                }
            ),
        )
        df = filter_teddymer_clusters(metadata, tmp_path / "out.tsv")
        assert df["domain1_chopping"].iloc[0] == "10-50"
        assert df["domain2_chopping"].iloc[0] == "100-200,250-300"

        # Downstream parser must be able to consume the strings.
        from teddympnn.data.teddymer import _parse_chopping

        assert _parse_chopping(df["domain1_chopping"].iloc[0]) == [(10, 50)]
        assert _parse_chopping(df["domain2_chopping"].iloc[0]) == [
            (100, 200),
            (250, 300),
        ]

    def test_built_from_start_end_columns(self, tmp_path: Path):
        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "uniprot_id": ["P00001"],
                    "domain1_start": [10],
                    "domain1_end": [50],
                    "domain2_start": [100],
                    "domain2_end": [200],
                }
            ),
        )
        df = filter_teddymer_clusters(metadata, tmp_path / "out.tsv")
        assert df["domain1_chopping"].iloc[0] == "10-50"
        assert df["domain2_chopping"].iloc[0] == "100-200"

    def test_joined_from_ted_table(self, tmp_path: Path):
        # Side-table mapping ted_id → chopping.
        ted_table = tmp_path / "ted_domains.tsv"
        pd.DataFrame(
            {
                "ted_id": [
                    "AF-Q9Y6K1-F1-model_v4_TED01",
                    "AF-Q9Y6K1-F1-model_v4_TED02",
                ],
                "chopping": ["10-80", "120-200"],
            }
        ).to_csv(ted_table, sep="\t", index=False)

        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "ted_id1": ["AF-Q9Y6K1-F1-model_v4_TED01"],
                    "ted_id2": ["AF-Q9Y6K1-F1-model_v4_TED02"],
                }
            ),
        )
        df = filter_teddymer_clusters(metadata, tmp_path / "out.tsv")
        assert df["domain1_chopping"].iloc[0] == "10-80"
        assert df["domain2_chopping"].iloc[0] == "120-200"
        # uniprot_id derived from the TED id.
        assert df["uniprot_id"].iloc[0] == "Q9Y6K1"

    def test_ted_join_drops_unresolved_rows(self, tmp_path: Path):
        ted_table = tmp_path / "ted_domains.tsv"
        pd.DataFrame(
            {
                "ted_id": ["AF-AAA-F1-model_v4_TED01"],
                "chopping": ["1-50"],
            }
        ).to_csv(ted_table, sep="\t", index=False)

        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0, 80.0],
                    "AvgIntPAE": [5.0, 5.0],
                    "InterfaceLength": [15, 15],
                    "ted_id1": [
                        "AF-AAA-F1-model_v4_TED01",
                        "AF-BBB-F1-model_v4_TED01",  # not in side table
                    ],
                    "ted_id2": [
                        "AF-AAA-F1-model_v4_TED01",
                        "AF-BBB-F1-model_v4_TED02",
                    ],
                }
            ),
        )
        df = filter_teddymer_clusters(metadata, tmp_path / "out.tsv")
        assert len(df) == 1
        assert df["uniprot_id"].iloc[0] == "AAA"

    def test_raises_when_chopping_unresolvable(self, tmp_path: Path):
        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "uniprot_id": ["P00001"],
                }
            ),
        )
        with pytest.raises(ValueError, match="Cannot derive"):
            filter_teddymer_clusters(metadata, tmp_path / "out.tsv")

    def test_require_chopping_false_warns(self, tmp_path: Path, caplog):
        metadata = self._write_metadata(
            tmp_path,
            pd.DataFrame(
                {
                    "InterfacePlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "uniprot_id": ["P00001"],
                }
            ),
        )
        df = filter_teddymer_clusters(
            metadata, tmp_path / "out.tsv", require_chopping=False
        )
        assert "domain1_chopping" not in df.columns
