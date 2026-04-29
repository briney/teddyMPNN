"""Tests for teddymer data acquisition pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used in fixture type annotations
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

from teddympnn.data import teddymer as teddymer_module
from teddympnn.data.teddymer import (
    _assemble_one,
    _parse_chopping,
    extract_and_assemble_dimers,
    filter_teddymer_clusters,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

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
                "AvgIntPlddt": [80.0, 60.0, 75.0, 90.0, 50.0],
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

        # Only entries with AvgIntPlddt > 70
        assert all(df["avgintplddt"] > 70.0)

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
                    "AvgIntPlddt": [80.0],
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
                    "AvgIntPlddt": [80.0],
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
                    "AvgIntPlddt": [80.0],
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
                    "AvgIntPlddt": [80.0, 80.0],
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
                    "AvgIntPlddt": [80.0],
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
                    "AvgIntPlddt": [80.0],
                    "AvgIntPAE": [5.0],
                    "InterfaceLength": [15],
                    "uniprot_id": ["P00001"],
                }
            ),
        )
        df = filter_teddymer_clusters(metadata, tmp_path / "out.tsv", require_chopping=False)
        assert "domain1_chopping" not in df.columns


# ---------------------------------------------------------------------------
# FoldSeek-driven dimer extraction
# ---------------------------------------------------------------------------


def _write_dummy_foldseek_db(prefix: Path) -> None:
    """Create empty sibling files for a fake FoldSeek DB at ``prefix``."""
    for suffix in ("", "_ca", "_h", "_ss", ".dbtype", ".index", ".lookup", ".source"):
        prefix.with_name(prefix.name + suffix).write_bytes(b"")


def _write_min_pdb(path: Path, chain_id: str, start_resnum: int, n_residues: int = 5) -> None:
    """Write a minimal valid PDB with a single chain of glycines (CA only)."""
    lines: list[str] = []
    atom_serial = 1
    for i in range(n_residues):
        resnum = start_resnum + i
        # 11 cols of the PDB ATOM format, fixed-width.
        lines.append(
            f"ATOM  {atom_serial:>5d}  CA  GLY {chain_id}{resnum:>4d}    "
            f"{i * 3.8:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C\n"
        )
        atom_serial += 1
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


class TestExtractAndAssembleDimers:
    def _manifest(self, tmp_path: Path, rows: Iterable[dict[str, object]]) -> Path:
        path = tmp_path / "filtered_manifest.tsv"
        pd.DataFrame(list(rows)).to_csv(path, sep="\t", index=False)
        return path

    def test_raises_when_foldseek_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        manifest = self._manifest(
            tmp_path,
            [{"dimerindex": 1, "ted_id1": "X_TED01", "ted_id2": "X_TED02"}],
        )
        monkeypatch.setattr(teddymer_module.shutil, "which", lambda _: None)
        with pytest.raises(RuntimeError, match="foldseek"):
            extract_and_assemble_dimers(manifest, tmp_path / "fake_db", tmp_path / "dimers")

    def test_raises_when_db_incomplete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        manifest = self._manifest(
            tmp_path,
            [{"dimerindex": 1, "ted_id1": "X_TED01", "ted_id2": "X_TED02"}],
        )
        # Only the main file exists; siblings are missing.
        db = tmp_path / "incomplete_db"
        db.write_bytes(b"")
        monkeypatch.setattr(teddymer_module.shutil, "which", lambda _: "/fake/foldseek")
        with pytest.raises(FileNotFoundError, match="incomplete"):
            extract_and_assemble_dimers(manifest, db, tmp_path / "dimers")

    def test_invokes_foldseek_per_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"dimerindex": i, "ted_id1": f"E{i}_TED01", "ted_id2": f"E{i}_TED02"} for i in (1, 2, 3)
        ]
        manifest = self._manifest(tmp_path, rows)
        db = tmp_path / "db" / "ted_db"
        db.parent.mkdir()
        _write_dummy_foldseek_db(db)
        monkeypatch.setattr(teddymer_module.shutil, "which", lambda _: "/fake/foldseek")

        run_mock = MagicMock()
        seen_id_files: list[Path] = []

        def fake_run(args: list[str], *, foldseek_binary: str = "foldseek") -> None:
            run_mock(args)
            if args[0] == "createsubdb":
                # createsubdb args: [createsubdb, --id-mode, 1, --subdb-mode, 1,
                #                    <ids>, <src>, <dst>]
                seen_id_files.append(Path(args[5]))
            elif args[0] == "convert2pdb":
                # convert2pdb args: [convert2pdb, --pdb-output-mode, 1,
                #                    <subset_db>, <pdb_dir>, ...]
                pdb_dir = Path(args[4])
                pdb_dir.mkdir(parents=True, exist_ok=True)
                for entry in seen_id_files[-1].read_text().splitlines():
                    entry = entry.strip()
                    if entry:
                        _write_min_pdb(pdb_dir / f"{entry}.pdb", "X", start_resnum=10)

        monkeypatch.setattr(teddymer_module, "_run_foldseek", fake_run)

        success = extract_and_assemble_dimers(
            manifest,
            db,
            tmp_path / "dimers",
            chunk_size=2,
            workers=1,
            keep_scratch=False,
        )

        assert success == 3
        # 2 chunks → 2 createsubdb + 2 convert2pdb invocations.
        assert run_mock.call_count == 4
        ops = [call.args[0][0] for call in run_mock.call_args_list]
        assert ops.count("createsubdb") == 2
        assert ops.count("convert2pdb") == 2

        # Output files exist with stable dimerindex naming.
        for i in (1, 2, 3):
            assert (tmp_path / "dimers" / f"{i}.pdb").exists()

        # Manifest written with expected schema.
        manifest_out = tmp_path / "dimers" / "manifest.tsv"
        assert manifest_out.exists()
        out_df = pd.read_csv(manifest_out, sep="\t")
        assert set(out_df.columns) >= {
            "structure_path",
            "chain_A",
            "chain_B",
            "source",
            "source_id",
            "split_group",
            "interface_residues",
        }
        assert (out_df["chain_A"] == "A").all()
        assert (out_df["chain_B"] == "B").all()

    def test_assemble_one_renames_and_renumbers(self, tmp_path: Path):
        ted1 = tmp_path / "ted1.pdb"
        ted2 = tmp_path / "ted2.pdb"
        out = tmp_path / "dimer.pdb"
        # Source files use chain X and start at residue 50 — the assembler
        # must rename to A/B and renumber from 1.
        _write_min_pdb(ted1, "X", start_resnum=50, n_residues=4)
        _write_min_pdb(ted2, "X", start_resnum=50, n_residues=6)

        assert _assemble_one(ted1, ted2, out) is True
        assert out.exists()

        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("dimer", str(out))
        chain_ids = [c.id for c in structure[0].get_chains()]
        assert chain_ids == ["A", "B"]
        chain_a_resnums = [r.id[1] for r in structure[0]["A"]]
        chain_b_resnums = [r.id[1] for r in structure[0]["B"]]
        assert chain_a_resnums == [1, 2, 3, 4]
        assert chain_b_resnums == [1, 2, 3, 4, 5, 6]

    def test_resumes_skipping_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"dimerindex": 1, "ted_id1": "E1_TED01", "ted_id2": "E1_TED02"},
            {"dimerindex": 2, "ted_id1": "E2_TED01", "ted_id2": "E2_TED02"},
        ]
        manifest = self._manifest(tmp_path, rows)
        db = tmp_path / "db" / "ted_db"
        db.parent.mkdir()
        _write_dummy_foldseek_db(db)
        monkeypatch.setattr(teddymer_module.shutil, "which", lambda _: "/fake/foldseek")

        # Pre-create row 1's output so resume should skip it.
        out_dir = tmp_path / "dimers"
        out_dir.mkdir()
        (out_dir / "1.pdb").write_text("ATOM\nEND\n")

        captured_ids: list[list[str]] = []

        def fake_run(args: list[str], *, foldseek_binary: str = "foldseek") -> None:
            if args[0] == "createsubdb":
                ids = [
                    line.strip() for line in Path(args[5]).read_text().splitlines() if line.strip()
                ]
                captured_ids.append(ids)
            elif args[0] == "convert2pdb":
                pdb_dir = Path(args[4])
                pdb_dir.mkdir(parents=True, exist_ok=True)
                for entry in captured_ids[-1]:
                    _write_min_pdb(pdb_dir / f"{entry}.pdb", "X", 10)

        monkeypatch.setattr(teddymer_module, "_run_foldseek", fake_run)

        extract_and_assemble_dimers(manifest, db, out_dir, chunk_size=10, workers=1)

        # Exactly one createsubdb invocation, and its ids list must exclude row 1's TEDs.
        assert len(captured_ids) == 1
        ids = captured_ids[0]
        assert "E1_TED01" not in ids
        assert "E1_TED02" not in ids
        assert "E2_TED01" in ids
        assert "E2_TED02" in ids
