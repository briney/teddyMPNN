"""Tests for the Teddymer full-atom reconstruction pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp
import pandas as pd
import pytest

from teddympnn.data import teddymer as teddymer_module
from teddympnn.data.teddymer import (
    TeddymerPrepareConfig,
    assemble_ted_domain_pdbs,
    build_teddymer_indices,
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
    """Create a minimal extracted Teddymer archive fixture.

    Uses the real archive metadata format: ``UniProtID`` is the full AlphaFold
    model stem (``AF-<acc>-F1-model``), not a bare accession. Values are drawn
    from real ``nonsingletonrep_metadata.tsv`` rows.
    """
    extracted = root / "raw"
    extracted.mkdir()

    pd.DataFrame(
        {
            "DimerIndex": [655, 1813],
            "UniProtID": ["AF-A0A009ESU5-F1-model", "AF-A0A009F653-F1-model"],
            "DomainPair": ["TED01:TED02", "TED01:TED02"],
            "MemberCount": [3, 2],
            "InterfaceLength": [57, 13],
            "AvgIntPAE": [7.003, 3.420],
            "AvgIntPlddt": [69.8246, 84.6154],
            "IntPlddt": ["5556:9999", "8888:9998"],
        }
    ).to_csv(extracted / "nonsingletonrep_metadata.tsv", sep="\t", index=False)
    (extracted / "cluster.tsv").write_text("cluster\trep\n1\t1\n")
    return extracted


class TestBuildTeddymerIndices:
    def test_builds_representatives_index_from_metadata(self, tmp_path: Path) -> None:
        extracted = _write_teddymer_fixture(tmp_path)

        indices = build_teddymer_indices(extracted, tmp_path / "prepared")

        reps = pd.read_csv(indices.representatives_path, sep="\t")
        assert len(reps) == 2
        assert indices.metadata_path.exists()
        assert indices.cluster_path is not None
        assert indices.cluster_path.exists()

        row = reps[reps["dimer_index"] == 655].iloc[0]
        # The TED id must be built from the AF model stem WITHOUT re-wrapping it
        # in another ``AF-...-F1-model`` (the original double-prefix bug).
        assert row["domain_a_ted_id"] == "AF-A0A009ESU5-F1-model_v4_TED01"
        assert row["domain_b_ted_id"] == "AF-A0A009ESU5-F1-model_v4_TED02"
        assert row["uniprot_id"] == "A0A009ESU5"
        assert row["domain_pair"] == "TED01:TED02"
        assert row["member_count"] == 3
        assert row["interface_residues"] == 57

    def test_respects_limit(self, tmp_path: Path) -> None:
        extracted = _write_teddymer_fixture(tmp_path)

        indices = build_teddymer_indices(extracted, tmp_path / "prepared", limit=1)

        reps = pd.read_csv(indices.representatives_path, sep="\t")
        assert len(reps) == 1

    def test_raises_on_missing_metadata_columns(self, tmp_path: Path) -> None:
        extracted = tmp_path / "raw"
        extracted.mkdir()
        pd.DataFrame({"UniProtID": ["AF-A0A005-F1-model"]}).to_csv(
            extracted / "nonsingletonrep_metadata.tsv", sep="\t", index=False
        )

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
                "rep_id": ["655", "1813"],
                "dimer_index": ["655", "1813"],
                "uniprot_id": ["A0A009ESU5", "A0A009F653"],
                "domain_pair": ["TED01:TED02", "TED01:TED02"],
                "domain_a_ted_id": [
                    "AF-A0A009ESU5-F1-model_v4_TED01",
                    "AF-A0A009F653-F1-model_v4_TED01",
                ],
                "domain_b_ted_id": [
                    "AF-A0A009ESU5-F1-model_v4_TED02",
                    "AF-A0A009F653-F1-model_v4_TED02",
                ],
                "member_count": [3, 2],
                "interface_residues": [57, 13],
                "avg_int_pae": [7.003, 3.420],
                "avg_int_plddt": [69.8246, 84.6154],
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
        assert (tmp_path / "dimers" / "655.pdb").exists()

        manifest = pd.read_csv(result.manifest_path, sep="\t")
        assert set(manifest["chain_A"]) == {"A"}
        assert set(manifest["chain_B"]) == {"B"}
        assert set(manifest["source"]) == {"teddymer"}
        assert set(manifest["source_id"].astype(str)) == {"655", "1813"}

    def test_resumes_existing_complete_pdb(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index = self._write_index(tmp_path / "index.tsv")
        out_dir = tmp_path / "dimers"
        out_dir.mkdir()
        (out_dir / "655.pdb").write_text("ATOM      1  CA  GLY A   1\nEND\n")
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
            if "A0A009F653" in ted_id:
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
        assert failure_df["rep_id"].astype(str).tolist() == ["1813"]
        assert "not found" in failure_df["error"].iloc[0]


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

        assert result.dimers == 2
        assert result.failures == 0
        assert result.metadata_path.exists()
        assert result.manifest_path.exists()


class _FakeResponse:
    """Minimal stand-in for an aiohttp response."""

    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text


class _FakeGet:
    """Async context manager mimicking ``session.get(...)``."""

    def __init__(self, item: _FakeResponse | Exception) -> None:
        self._item = item

    async def __aenter__(self) -> _FakeResponse:
        if isinstance(self._item, Exception):
            raise self._item
        return self._item

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeSession:
    """Yields queued responses/exceptions, repeating the last once exhausted."""

    def __init__(self, items: list[_FakeResponse | Exception]) -> None:
        self._items = list(items)
        self._current: _FakeResponse | Exception | None = None
        self.calls = 0

    def get(self, _url: str, headers: dict[str, str] | None = None) -> _FakeGet:
        self.calls += 1
        if self._items:
            self._current = self._items.pop(0)
        assert self._current is not None
        return _FakeGet(self._current)


class TestFetchDomainPdb:
    @staticmethod
    def _config() -> TeddymerPrepareConfig:
        return TeddymerPrepareConfig(retries=3)

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr(teddymer_module.asyncio, "sleep", _instant)

    @pytest.mark.parametrize("status", [404, 500])
    def test_http_error_status_is_not_retried(self, status: int) -> None:
        session = _FakeSession([_FakeResponse(status, "nope")])

        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            asyncio.run(
                teddymer_module._fetch_domain_pdb(
                    session, "AF-A0A009ESU5-F1-model_v4_TED01", self._config()
                )
            )

        # A definitive HTTP status is not retried.
        assert session.calls == 1

    def test_transient_error_is_retried_then_succeeds(self) -> None:
        pdb = _min_full_atom_pdb("Z", 1)
        session = _FakeSession([aiohttp.ClientError("connection reset"), _FakeResponse(200, pdb)])

        result = asyncio.run(
            teddymer_module._fetch_domain_pdb(
                session, "AF-A0A009ESU5-F1-model_v4_TED01", self._config()
            )
        )

        assert result == pdb
        assert session.calls == 2

    @pytest.mark.parametrize("status", [429, 503])
    def test_transient_http_status_is_retried_then_succeeds(self, status: int) -> None:
        # 429/5xx-gateway statuses are transient (e.g. TED rate-limiting under
        # concurrent load) and must be retried, unlike the definitive 404/500.
        pdb = _min_full_atom_pdb("Z", 1)
        session = _FakeSession([_FakeResponse(status, "busy"), _FakeResponse(200, pdb)])

        result = asyncio.run(
            teddymer_module._fetch_domain_pdb(
                session, "AF-A0A009ESU5-F1-model_v4_TED01", self._config()
            )
        )

        assert result == pdb
        assert session.calls == 2

    def test_transient_error_exhausts_retries(self) -> None:
        session = _FakeSession([aiohttp.ClientError("connection reset")])

        with pytest.raises(RuntimeError, match="after 3 attempts"):
            asyncio.run(
                teddymer_module._fetch_domain_pdb(
                    session, "AF-A0A009ESU5-F1-model_v4_TED01", self._config()
                )
            )

        assert session.calls == 3
