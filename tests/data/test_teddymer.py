"""Tests for the Teddymer full-atom reconstruction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from teddympnn.data import teddymer as teddymer_module
from teddympnn.data.teddymer import (
    TeddymerPrepareConfig,
    assemble_ted_domain_pdbs,
    build_teddymer_indices,
    link_nonsingleton_subset,
    prepare_teddymer_data,
    reconstruct_teddymer_dimers,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_min_full_atom_pdb(path: Path, chain_id: str, start_resnum: int) -> None:
    """Write a tiny full-atom-ish PDB with side-chain atoms."""
    path.write_text(_min_full_atom_pdb(chain_id, start_resnum))


def _min_full_atom_pdb(chain_id: str, start_resnum: int) -> str:
    """Return a tiny PDB string with two residues and side-chain atoms."""
    lines: list[str] = []
    serial = 1
    atoms = ("N", "CA", "C", "O", "CB")
    for offset, resname in enumerate(("ALA", "LY")):
        residue = "GLY" if resname == "LY" else resname
        resnum = start_resnum + offset
        for atom_idx, atom in enumerate(atoms):
            element = atom[0]
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain_id}{resnum:4d}    "
                f"{float(offset):8.3f}{float(atom_idx):8.3f}{0.0:8.3f}"
                f"  1.00 50.00           {element:>2s}\n"
            )
            serial += 1
    lines.append("TER\nEND\n")
    return "".join(lines)


def _write_teddymer_fixture(root: Path) -> Path:
    """Create a minimal extracted Teddymer archive fixture."""
    extracted = root / "raw"
    extracted.mkdir()

    pd.DataFrame(
        {
            "DimerIndex": [1, 3],
            "UniProtID": ["A0A005", "Q9ABC1"],
            "DomainPair": ["TED01:TED02", "TED01:TED02"],
            "MemberCount": [5, 2],
            "InterfaceLength": [15, 20],
            "AvgIntPAE": [4.5, 6.0],
            "AvgIntPlddt": [85.0, 91.0],
            "IntPlddt": ["80,85:90,92", "88,89:93,94"],
        }
    ).to_csv(extracted / "nonsingletonrep_metadata.tsv", sep="\t", index=False)
    (extracted / "cluster.tsv").write_text("cluster\trep\n1\t1\n")

    dimer_dir = extracted / "dir_ted_afdb50_cath_dimerdb"
    dimer_dir.mkdir()
    (dimer_dir / "ted_afdb50_cath_dimerdb.source").write_text(
        "\n".join(
            [
                "1DI_A0A005_v4_TED01\t1DI_A0A005_v4_TED02",
                "2DI_B0B000_v4_TED03\t2DI_B0B000_v4_TED04",
                "3DI_Q9ABC1_v4_TED01\t3DI_Q9ABC1_v4_TED02",
            ]
        )
        + "\n"
    )

    rep_dir = extracted / "dir_teddymer_repdb"
    rep_dir.mkdir()
    (rep_dir / "teddymer_repdb.source").write_text("rep_a\t1\nrep_b\t2\nrep_c\t3\n")
    return extracted


class TestBuildTeddymerIndices:
    def test_builds_all_and_nonsingleton_indices(self, tmp_path: Path) -> None:
        extracted = _write_teddymer_fixture(tmp_path)

        indices = build_teddymer_indices(extracted, tmp_path / "prepared")

        all_df = pd.read_csv(indices.all_representatives_path, sep="\t")
        non_df = pd.read_csv(indices.nonsingleton_representatives_path, sep="\t")

        assert len(all_df) == 3
        assert len(non_df) == 2
        assert indices.metadata_path.exists()
        assert indices.cluster_path is not None
        assert indices.cluster_path.exists()

        row = non_df[non_df["dimer_index"] == 1].iloc[0]
        assert row["domain_a_ted_id"] == "AF-A0A005-F1-model_v4_TED01"
        assert row["domain_b_ted_id"] == "AF-A0A005-F1-model_v4_TED02"
        assert row["interface_residues"] == 15

        all_row = all_df[all_df["dimer_index"] == 2].iloc[0]
        assert all_row["rep_id"] == "rep_b"
        assert all_row["domain_pair"] == "TED03:TED04"

    def test_raises_on_missing_metadata_columns(self, tmp_path: Path) -> None:
        extracted = tmp_path / "raw"
        extracted.mkdir()
        pd.DataFrame({"UniProtID": ["A0A005"]}).to_csv(
            extracted / "nonsingletonrep_metadata.tsv", sep="\t", index=False
        )
        (extracted / "ted_afdb50_cath_dimerdb.source").write_text("")
        (extracted / "teddymer_repdb.source").write_text("")

        with pytest.raises(ValueError, match="missing required"):
            build_teddymer_indices(extracted, tmp_path / "prepared")


class TestPdbAssembly:
    def test_assembles_full_atom_domains_as_chains_a_b(self) -> None:
        assembled = assemble_ted_domain_pdbs(
            _min_full_atom_pdb("X", 50),
            _min_full_atom_pdb("Y", 200),
        )

        atom_lines = [line for line in assembled.splitlines() if line.startswith("ATOM")]
        assert len(atom_lines) == 20
        assert atom_lines[0][21] == "A"
        assert atom_lines[0][22:26].strip() == "1"
        assert atom_lines[9][22:26].strip() == "2"
        assert atom_lines[10][21] == "B"
        assert atom_lines[10][22:26].strip() == "1"
        assert any(line[12:16].strip() == "CB" for line in atom_lines)
        assert assembled.endswith("END\n")

    def test_raises_when_domain_has_no_atoms(self) -> None:
        with pytest.raises(ValueError, match="ATOM"):
            assemble_ted_domain_pdbs("HEADER empty\nEND\n", _min_full_atom_pdb("Y", 1))


class TestReconstructTeddymerDimers:
    def _write_index(self, path: Path) -> Path:
        pd.DataFrame(
            {
                "rep_id": ["rep_a", "rep_b"],
                "dimer_index": ["1", "2"],
                "uniprot_id": ["A0A005", "B0B000"],
                "domain_pair": ["TED01:TED02", "TED03:TED04"],
                "domain_a_ted_id": [
                    "AF-A0A005-F1-model_v4_TED01",
                    "AF-B0B000-F1-model_v4_TED03",
                ],
                "domain_b_ted_id": [
                    "AF-A0A005-F1-model_v4_TED02",
                    "AF-B0B000-F1-model_v4_TED04",
                ],
                "member_count": [5, None],
                "interface_residues": [15, None],
                "avg_int_pae": [4.5, None],
                "avg_int_plddt": [85.0, None],
            }
        ).to_csv(path, sep="\t", index=False)
        return path

    def test_reconstructs_with_mocked_ted_downloads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index = self._write_index(tmp_path / "index.tsv")
        seen: list[str] = []

        async def fake_fetch(_session: Any, ted_id: str, _config: TeddymerPrepareConfig) -> str:
            seen.append(ted_id)
            return _min_full_atom_pdb("Z", 100)

        monkeypatch.setattr(teddymer_module, "_fetch_domain_pdb", fake_fetch)

        result = reconstruct_teddymer_dimers(
            index,
            tmp_path / "dimers",
            TeddymerPrepareConfig(output_dir=tmp_path, workers=2),
        )

        assert result.success_count == 2
        assert result.failure_count == 0
        assert len(seen) == 4
        assert (tmp_path / "dimers" / "rep_a.pdb").exists()

        manifest = pd.read_csv(result.manifest_path, sep="\t")
        assert set(manifest["chain_A"]) == {"A"}
        assert set(manifest["chain_B"]) == {"B"}
        assert set(manifest["source"]) == {"teddymer"}
        assert set(manifest["source_id"]) == {"rep_a", "rep_b"}

    def test_resumes_existing_complete_pdb(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index = self._write_index(tmp_path / "index.tsv")
        out_dir = tmp_path / "dimers"
        out_dir.mkdir()
        (out_dir / "rep_a.pdb").write_text("ATOM      1  CA  GLY A   1\nEND\n")
        seen: list[str] = []

        async def fake_fetch(_session: Any, ted_id: str, _config: TeddymerPrepareConfig) -> str:
            seen.append(ted_id)
            return _min_full_atom_pdb("Z", 100)

        monkeypatch.setattr(teddymer_module, "_fetch_domain_pdb", fake_fetch)

        result = reconstruct_teddymer_dimers(
            index,
            out_dir,
            TeddymerPrepareConfig(output_dir=tmp_path, workers=2),
        )

        assert result.success_count == 2
        assert len(seen) == 2

    def test_logs_download_failures(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index = self._write_index(tmp_path / "index.tsv")

        async def fake_fetch(_session: Any, ted_id: str, _config: TeddymerPrepareConfig) -> str:
            if "B0B000" in ted_id:
                raise RuntimeError("not found")
            return _min_full_atom_pdb("Z", 100)

        monkeypatch.setattr(teddymer_module, "_fetch_domain_pdb", fake_fetch)

        failures = tmp_path / "logs" / "failures.tsv"
        result = reconstruct_teddymer_dimers(
            index,
            tmp_path / "dimers",
            TeddymerPrepareConfig(output_dir=tmp_path, workers=2),
            failures_path=failures,
        )

        assert result.success_count == 1
        assert result.failure_count == 1
        failure_df = pd.read_csv(failures, sep="\t")
        assert failure_df["rep_id"].tolist() == ["rep_b"]
        assert "not found" in failure_df["error"].iloc[0]


class TestNonsingletonSubset:
    def test_links_subset_and_writes_manifest(self, tmp_path: Path) -> None:
        all_dir = tmp_path / "all"
        all_dir.mkdir()
        _write_min_full_atom_pdb(all_dir / "rep_a.pdb", "A", 1)
        _write_min_full_atom_pdb(all_dir / "rep_b.pdb", "A", 1)

        all_manifest = all_dir / "manifest.tsv"
        pd.DataFrame(
            {
                "structure_path": [str(all_dir / "rep_a.pdb"), str(all_dir / "rep_b.pdb")],
                "chain_A": ["A", "A"],
                "chain_B": ["B", "B"],
                "source": ["teddymer", "teddymer"],
                "source_id": ["rep_a", "rep_b"],
                "split_group": ["rep_a", "rep_b"],
                "interface_residues": [15, 0],
                "rep_id": ["rep_a", "rep_b"],
                "dimer_index": ["1", "2"],
            }
        ).to_csv(all_manifest, sep="\t", index=False)

        nonsingleton = tmp_path / "nonsingleton.tsv"
        pd.DataFrame(
            {
                "rep_id": ["rep_a"],
                "dimer_index": ["1"],
                "uniprot_id": ["A0A005"],
                "domain_pair": ["TED01:TED02"],
                "domain_a_ted_id": ["AF-A0A005-F1-model_v4_TED01"],
                "domain_b_ted_id": ["AF-A0A005-F1-model_v4_TED02"],
                "member_count": [5],
                "interface_residues": [15],
                "avg_int_pae": [4.5],
                "avg_int_plddt": [85.0],
            }
        ).to_csv(nonsingleton, sep="\t", index=False)

        manifest_path = link_nonsingleton_subset(
            all_manifest,
            nonsingleton,
            tmp_path / "nonsingleton_dimers",
        )

        manifest = pd.read_csv(manifest_path, sep="\t")
        assert manifest["source_id"].tolist() == ["rep_a"]
        subset_pdb = tmp_path / "nonsingleton_dimers" / "rep_a.pdb"
        assert subset_pdb.exists()


class TestPrepareTeddymerData:
    def test_prepare_end_to_end_with_mocked_downloads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        extracted = _write_teddymer_fixture(tmp_path)

        def fake_download(_config: TeddymerPrepareConfig) -> Path:
            return extracted

        async def fake_fetch(_session: Any, _ted_id: str, _config: TeddymerPrepareConfig) -> str:
            return _min_full_atom_pdb("Z", 100)

        monkeypatch.setattr(teddymer_module, "download_and_extract_teddymer", fake_download)
        monkeypatch.setattr(teddymer_module, "_fetch_domain_pdb", fake_fetch)

        result = prepare_teddymer_data(TeddymerPrepareConfig(output_dir=tmp_path / "prepared"))

        assert result.all_dimers == 3
        assert result.nonsingleton_dimers == 2
        assert result.failures == 0
        assert result.metadata_path.exists()
        assert result.all_manifest_path.exists()
        assert result.nonsingleton_manifest_path.exists()
