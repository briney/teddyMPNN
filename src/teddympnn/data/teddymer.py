"""Teddymer full-atom dimer data acquisition pipeline.

The Teddymer archive ships metadata plus C-alpha-only FoldSeek databases. For
training, teddyMPNN needs full side-chain PDBs, so this module follows the
published Teddymer reconstruction recipe: use the metadata/source indices to
identify the two TED domains for each representative dimer, download the
corresponding full TED-domain PDBs from TED, and assemble them as chains A/B.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)

TEDDYMER_ARCHIVE_URL = "https://teddymer.steineggerlab.workers.dev/foldseek/teddymer.tar.gz"
TED_DOMAIN_URL_TEMPLATE = "https://ted.cathdb.info/api/v1/files/{ted_id}.pdb"

METADATA_FILENAME = "nonsingletonrep_metadata.tsv"
CLUSTER_FILENAME = "cluster.tsv"
REPRESENTATIVES_FILENAME = "representatives.tsv"
FAILURES_FILENAME = "failures.tsv"

# AlphaFold DB model version embedded in TED domain identifiers.
TED_MODEL_VERSION = "v4"

TRAINING_MANIFEST_COLUMNS: tuple[str, ...] = (
    "structure_path",
    "chain_A",
    "chain_B",
    "source",
    "source_id",
    "split_group",
    "interface_residues",
)

INDEX_COLUMNS: tuple[str, ...] = (
    "rep_id",
    "dimer_index",
    "uniprot_id",
    "domain_pair",
    "domain_a_ted_id",
    "domain_b_ted_id",
    "member_count",
    "interface_residues",
    "avg_int_pae",
    "avg_int_plddt",
)

_AF_STEM_RE = re.compile(r"AF-(?P<acc>[A-Za-z0-9]+)-F\d+-model", re.IGNORECASE)
_TED_DOMAIN_RE = re.compile(r"TED\d+", re.IGNORECASE)
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TeddymerPrepareConfig:
    """Configuration for the full Teddymer preparation workflow.

    Args:
        output_dir: Root directory for downloaded metadata and reconstructed PDBs.
        archive_url: URL for ``teddymer.tar.gz``.
        workers: Number of concurrent TED-domain HTTP requests.
        retries: Number of attempts per TED-domain request.
        timeout_seconds: Per-request timeout.
        overwrite: Rebuild existing dimer PDBs and manifests.
        keep_archive: Keep ``teddymer.tar.gz`` after extraction.
        domain_cache_dir: Optional cache for downloaded TED-domain PDBs. Leave
            unset to avoid storing one extra full copy of the domain data.
        limit: Reconstruct at most this many representative dimers. Useful for
            pilot/smoke runs; leave unset to process the full set.
    """

    output_dir: Path = Path("data/teddymer")
    archive_url: str = TEDDYMER_ARCHIVE_URL
    workers: int = 8
    retries: int = 3
    timeout_seconds: float = 60.0
    overwrite: bool = False
    keep_archive: bool = True
    domain_cache_dir: Path | None = None
    limit: int | None = None


@dataclass(frozen=True)
class TeddymerDimerRecord:
    """Normalized Teddymer representative dimer record."""

    rep_id: str
    dimer_index: str
    uniprot_id: str
    domain_pair: str
    domain_a_ted_id: str
    domain_b_ted_id: str
    member_count: int | None = None
    interface_residues: int | None = None
    avg_int_pae: float | None = None
    avg_int_plddt: float | None = None

    @property
    def output_stem(self) -> str:
        """Filesystem-safe stem for the reconstructed dimer PDB."""
        stem = self.rep_id or self.dimer_index
        return _SAFE_STEM_RE.sub("_", stem).strip("._") or f"dimer_{self.dimer_index}"

    def to_index_row(self) -> dict[str, str | int | float | None]:
        """Convert the record to a source-index TSV row."""
        return {
            "rep_id": self.rep_id,
            "dimer_index": self.dimer_index,
            "uniprot_id": self.uniprot_id,
            "domain_pair": self.domain_pair,
            "domain_a_ted_id": self.domain_a_ted_id,
            "domain_b_ted_id": self.domain_b_ted_id,
            "member_count": self.member_count,
            "interface_residues": self.interface_residues,
            "avg_int_pae": self.avg_int_pae,
            "avg_int_plddt": self.avg_int_plddt,
        }

    def to_manifest_row(self, structure_path: Path) -> dict[str, str | int | float | None]:
        """Convert the record to a training manifest row."""
        row = self.to_index_row()
        row.update(
            {
                "structure_path": str(structure_path),
                "chain_A": "A",
                "chain_B": "B",
                "source": "teddymer",
                "source_id": self.rep_id,
                "split_group": self.rep_id,
                "interface_residues": int(self.interface_residues or 0),
            }
        )
        return row


@dataclass(frozen=True)
class TeddymerIndices:
    """Paths to normalized Teddymer source indices."""

    metadata_path: Path
    cluster_path: Path | None
    representatives_path: Path


@dataclass(frozen=True)
class TeddymerReconstructionResult:
    """Summary of a dimer reconstruction pass."""

    manifest_path: Path
    failures_path: Path
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class TeddymerPrepareResult:
    """Summary of the complete Teddymer preparation workflow."""

    metadata_path: Path
    manifest_path: Path
    failures_path: Path
    dimers: int
    failures: int


def prepare_teddymer_data(config: TeddymerPrepareConfig | None = None) -> TeddymerPrepareResult:
    """Download, index, and reconstruct Teddymer full-atom dimer PDBs.

    Args:
        config: Workflow configuration. Defaults to ``TeddymerPrepareConfig()``.

    Returns:
        Paths and counts for the generated metadata, manifest, and failures.
    """
    config = config or TeddymerPrepareConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = download_and_extract_teddymer(config)
    indices = build_teddymer_indices(extracted_dir, output_dir, limit=config.limit)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures_path = logs_dir / FAILURES_FILENAME

    result = reconstruct_teddymer_dimers(
        indices.representatives_path,
        output_dir / "dimers",
        config,
        failures_path=failures_path,
    )

    return TeddymerPrepareResult(
        metadata_path=indices.metadata_path,
        manifest_path=result.manifest_path,
        failures_path=failures_path,
        dimers=result.success_count,
        failures=result.failure_count,
    )


def download_and_extract_teddymer(config: TeddymerPrepareConfig) -> Path:
    """Download ``teddymer.tar.gz`` and extract it into the output directory.

    Args:
        config: Workflow configuration.

    Returns:
        Directory containing the extracted Teddymer archive contents.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "teddymer.tar.gz"
    extracted_dir = output_dir / "raw"

    if not archive_path.exists() or config.overwrite:
        _download_with_progress(config.archive_url, archive_path)
    else:
        logger.info("Using existing Teddymer archive at %s", archive_path)

    if config.overwrite and extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    if not _contains_teddymer_inputs(extracted_dir):
        extracted_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s to %s", archive_path, extracted_dir)
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extracted_dir, filter="data")
    else:
        logger.info("Using existing extracted Teddymer data at %s", extracted_dir)

    if not config.keep_archive:
        archive_path.unlink(missing_ok=True)

    return extracted_dir


def build_teddymer_indices(
    extracted_dir: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> TeddymerIndices:
    """Build the normalized non-singleton representative index from metadata.

    The Teddymer archive's curated ``nonsingletonrep_metadata.tsv`` is the
    source of truth for the training set: each row is one non-singleton cluster
    representative with its UniProt model stem, TED domain pair, and interface
    metrics. The full ``.source``/``.lookup`` FoldSeek databases enumerate all
    ~10M dimers and are not used here.

    Args:
        extracted_dir: Directory containing extracted Teddymer archive contents.
        output_dir: Root output directory for normalized metadata files.
        limit: Keep at most this many records (pilot/smoke runs).

    Returns:
        Paths to generated metadata/index files.
    """
    extracted_dir = Path(extracted_dir)
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_src = _find_required_file(extracted_dir, METADATA_FILENAME)
    metadata_path = metadata_dir / METADATA_FILENAME
    shutil.copy2(metadata_src, metadata_path)

    cluster_path: Path | None = None
    cluster_src = _find_optional_file(extracted_dir, CLUSTER_FILENAME)
    if cluster_src is not None:
        cluster_path = metadata_dir / CLUSTER_FILENAME
        shutil.copy2(cluster_src, cluster_path)

    records = _records_from_metadata(metadata_path)
    records = [
        record for record in records if record.member_count is None or record.member_count > 1
    ]
    if limit is not None:
        records = records[:limit]

    representatives_path = metadata_dir / REPRESENTATIVES_FILENAME
    _write_records(records, representatives_path)
    logger.info("Wrote %d representative records to %s", len(records), representatives_path)

    return TeddymerIndices(
        metadata_path=metadata_path,
        cluster_path=cluster_path,
        representatives_path=representatives_path,
    )


def reconstruct_teddymer_dimers(
    index_path: str | Path,
    output_dir: str | Path,
    config: TeddymerPrepareConfig,
    *,
    failures_path: str | Path | None = None,
) -> TeddymerReconstructionResult:
    """Reconstruct full-atom Teddymer dimers from a normalized source index.

    Args:
        index_path: TSV produced by :func:`build_teddymer_indices`.
        output_dir: Directory for assembled dimer PDBs and ``manifest.tsv``.
        config: Workflow configuration.
        failures_path: Optional shared failure log path.

    Returns:
        Reconstruction summary.
    """
    index_path = Path(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = (
        Path(failures_path) if failures_path is not None else output_dir / FAILURES_FILENAME
    )

    records = _read_records(index_path)
    result = asyncio.run(_reconstruct_records_async(records, output_dir, config))

    manifest = pd.DataFrame(result["manifest_rows"])
    manifest_path = output_dir / "manifest.tsv"
    if not manifest.empty:
        leading = list(TRAINING_MANIFEST_COLUMNS)
        trailing = [column for column in manifest.columns if column not in leading]
        manifest = manifest[leading + trailing]
    else:
        manifest = pd.DataFrame(columns=list(TRAINING_MANIFEST_COLUMNS))
    manifest.to_csv(manifest_path, sep="\t", index=False)

    failures = result["failures"]
    if failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_df = pd.DataFrame(failures)
        if failures_path.exists():
            previous = pd.read_csv(failures_path, sep="\t")
            failures_df = pd.concat([previous, failures_df], ignore_index=True)
        failures_df.to_csv(failures_path, sep="\t", index=False)

    success_count = len(result["manifest_rows"])
    failure_count = len(failures)
    logger.info("Reconstructed %d / %d Teddymer dimers", success_count, len(records))
    if failure_count:
        logger.warning(
            "Wrote %d Teddymer reconstruction failures to %s",
            failure_count,
            failures_path,
        )

    return TeddymerReconstructionResult(
        manifest_path=manifest_path,
        failures_path=failures_path,
        success_count=success_count,
        failure_count=failure_count,
    )


def download_teddymer_metadata(output_dir: str | Path) -> Path:
    """Backward-compatible helper that downloads and extracts Teddymer metadata.

    Prefer :func:`prepare_teddymer_data` for the full reconstruction workflow.

    Args:
        output_dir: Directory in which to extract the Teddymer archive.

    Returns:
        Directory containing the extracted archive.
    """
    config = TeddymerPrepareConfig(output_dir=Path(output_dir))
    return download_and_extract_teddymer(config)


def _download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` with a rich progress bar and atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed HTTPS dataset URL
        total = int(resp.headers.get("Content-Length", 0)) or None
        columns = (
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*columns) as progress, tmp.open("wb") as handle:
            task_id = progress.add_task(dest.name, total=total)
            while chunk := resp.read(chunk_size):
                handle.write(chunk)
                progress.advance(task_id, len(chunk))
    tmp.replace(dest)


def _contains_teddymer_inputs(path: Path) -> bool:
    """Return True if an extracted archive directory contains required files."""
    return path.exists() and _find_optional_file(path, METADATA_FILENAME) is not None


def _find_required_file(root: Path, filename: str) -> Path:
    """Find a required file below ``root``."""
    path = _find_optional_file(root, filename)
    if path is None:
        msg = f"Could not find {filename!r} under {root}"
        raise FileNotFoundError(msg)
    return path


def _find_optional_file(root: Path, filename: str) -> Path | None:
    """Find an optional file below ``root`` by basename."""
    if not root.exists():
        return None
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def _records_from_metadata(metadata_path: Path) -> list[TeddymerDimerRecord]:
    """Parse ``nonsingletonrep_metadata.tsv`` into normalized dimer records."""
    df = pd.read_csv(metadata_path, sep="\t")
    columns = {column.strip().lower(): column for column in df.columns}
    required = ("dimerindex", "uniprotid", "domainpair")
    missing = [column for column in required if column not in columns]
    if missing:
        msg = f"{metadata_path} missing required Teddymer metadata columns: {missing}"
        raise ValueError(msg)

    records: list[TeddymerDimerRecord] = []
    for _, row in df.iterrows():
        dimer_index = str(row[columns["dimerindex"]])
        model_stem = str(row[columns["uniprotid"]])
        domain_a, domain_b = _split_domain_pair(str(row[columns["domainpair"]]))
        record = TeddymerDimerRecord(
            rep_id=dimer_index,
            dimer_index=dimer_index,
            uniprot_id=_bare_accession(model_stem),
            domain_pair=f"{domain_a}:{domain_b}",
            domain_a_ted_id=_ted_domain_id(model_stem, domain_a),
            domain_b_ted_id=_ted_domain_id(model_stem, domain_b),
            member_count=_optional_int(row, columns.get("membercount")),
            interface_residues=_optional_int(row, columns.get("interfacelength")),
            avg_int_pae=_optional_float(row, columns.get("avgintpae")),
            avg_int_plddt=_optional_float(row, columns.get("avgintplddt")),
        )
        records.append(record)
    return records


def _split_domain_pair(domain_pair: str) -> tuple[str, str]:
    """Split a Teddymer ``DomainPair`` value into two TED domain labels."""
    parts = [part.strip().upper() for part in re.split(r"[:;,|]\s*", domain_pair) if part.strip()]
    if len(parts) != 2 or not all(_TED_DOMAIN_RE.fullmatch(part) for part in parts):
        msg = f"Expected DomainPair with two TED domains, got {domain_pair!r}"
        raise ValueError(msg)
    return parts[0], parts[1]


def _ted_domain_id(model_stem: str, domain: str) -> str:
    """Build the TED API file stem from an AlphaFold model stem and TED domain.

    The metadata ``UniProtID`` is already the full AlphaFold model stem
    (e.g. ``AF-A0A009ESU5-F1-model``); the TED domain PDB id appends the model
    version and domain (``AF-A0A009ESU5-F1-model_v4_TED01``).
    """
    return f"{model_stem}_{TED_MODEL_VERSION}_{domain.upper()}"


def _bare_accession(model_stem: str) -> str:
    """Extract the bare UniProt accession from an AlphaFold model stem."""
    match = _AF_STEM_RE.match(model_stem)
    return match.group("acc") if match is not None else model_stem


def _optional_int(row: pd.Series, column: str | None) -> int | None:
    """Read an optional integer value from a DataFrame row."""
    if column is None or pd.isna(row[column]):
        return None
    return int(row[column])


def _optional_float(row: pd.Series, column: str | None) -> float | None:
    """Read an optional float value from a DataFrame row."""
    if column is None or pd.isna(row[column]):
        return None
    return float(row[column])


def _write_records(records: list[TeddymerDimerRecord], path: Path) -> None:
    """Write normalized dimer records to a TSV file."""
    df = pd.DataFrame([record.to_index_row() for record in records], columns=list(INDEX_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _read_records(path: Path) -> list[TeddymerDimerRecord]:
    """Read normalized dimer records from a TSV file."""
    df = pd.read_csv(path, sep="\t")
    return [_record_from_series(row) for _, row in df.iterrows()]


def _record_from_series(row: pd.Series) -> TeddymerDimerRecord:
    """Convert a normalized record row back into a dataclass."""
    return TeddymerDimerRecord(
        rep_id=str(row["rep_id"]),
        dimer_index=str(row["dimer_index"]),
        uniprot_id=str(row["uniprot_id"]),
        domain_pair=str(row["domain_pair"]),
        domain_a_ted_id=str(row["domain_a_ted_id"]),
        domain_b_ted_id=str(row["domain_b_ted_id"]),
        member_count=_series_optional_int(row, "member_count"),
        interface_residues=_series_optional_int(row, "interface_residues"),
        avg_int_pae=_series_optional_float(row, "avg_int_pae"),
        avg_int_plddt=_series_optional_float(row, "avg_int_plddt"),
    )


def _series_optional_int(row: pd.Series, column: str) -> int | None:
    """Read a nullable integer from a normalized record row."""
    if column not in row or pd.isna(row[column]):
        return None
    return int(row[column])


def _series_optional_float(row: pd.Series, column: str) -> float | None:
    """Read a nullable float from a normalized record row."""
    if column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


async def _reconstruct_records_async(
    records: list[TeddymerDimerRecord],
    output_dir: Path,
    config: TeddymerPrepareConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Download TED domains concurrently and assemble representative dimers."""
    connector = aiohttp.TCPConnector(limit=max(config.workers, 1))
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    semaphore = asyncio.Semaphore(max(config.workers, 1))
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with Progress() as progress:
            task_id = progress.add_task("Reconstructing Teddymer dimers", total=len(records))

            async def _run_one(record: TeddymerDimerRecord) -> None:
                async with semaphore:
                    try:
                        row = await _reconstruct_one_record(session, record, output_dir, config)
                        manifest_rows.append(row)
                    except Exception as exc:
                        failures.append(
                            {
                                "rep_id": record.rep_id,
                                "dimer_index": record.dimer_index,
                                "domain_a_ted_id": record.domain_a_ted_id,
                                "domain_b_ted_id": record.domain_b_ted_id,
                                "error": str(exc),
                            }
                        )
                    finally:
                        progress.advance(task_id)

            await asyncio.gather(*(_run_one(record) for record in records))

    manifest_rows.sort(key=lambda row: str(row["source_id"]))
    failures.sort(key=lambda row: row["rep_id"])
    return {"manifest_rows": manifest_rows, "failures": failures}


async def _reconstruct_one_record(
    session: aiohttp.ClientSession,
    record: TeddymerDimerRecord,
    output_dir: Path,
    config: TeddymerPrepareConfig,
) -> dict[str, Any]:
    """Download two TED-domain PDBs and write one assembled dimer PDB."""
    output_path = output_dir / f"{record.output_stem}.pdb"
    if output_path.exists() and not config.overwrite and _looks_like_complete_pdb(output_path):
        return record.to_manifest_row(output_path)

    pdb_a = await _fetch_domain_pdb(session, record.domain_a_ted_id, config)
    pdb_b = await _fetch_domain_pdb(session, record.domain_b_ted_id, config)
    assembled = assemble_ted_domain_pdbs(pdb_a, pdb_b)

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(assembled)
    tmp_path.replace(output_path)
    return record.to_manifest_row(output_path)


# Transient HTTP statuses worth retrying (rate-limiting / gateway hiccups under
# concurrent load). TED returns a deterministic 500 for structures purged in the
# AFDB v4->v6 transition, so 500 is treated as definitive and is NOT in this set.
_RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})


class _TedHttpStatusError(RuntimeError):
    """A definitive non-200 HTTP status from the TED API (e.g. 404/500).

    Definitive statuses are not retried; a structure purged in the AFDB v4->v6
    transition deterministically 500s, so retrying cannot help. Transient
    statuses (see ``_RETRYABLE_HTTP_STATUS``) raise a plain error instead and
    follow the normal retry path.
    """


async def _fetch_domain_pdb(
    session: aiohttp.ClientSession,
    ted_id: str,
    config: TeddymerPrepareConfig,
) -> str:
    """Fetch a TED-domain PDB, optionally using a local cache.

    Definitive HTTP statuses (e.g. 404/500) fail fast; transient errors
    (timeouts, connection resets, and ``_RETRYABLE_HTTP_STATUS`` responses such
    as 429/503) are retried with exponential backoff.
    """
    cache_path: Path | None = None
    if config.domain_cache_dir is not None:
        cache_dir = Path(config.domain_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_SAFE_STEM_RE.sub('_', ted_id)}.pdb"
        if cache_path.exists() and _looks_like_complete_pdb(cache_path):
            return cache_path.read_text()

    url = TED_DOMAIN_URL_TEMPLATE.format(ted_id=ted_id)
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
                if response.status != 200:
                    text: str = await response.text()
                    msg = f"GET {url} returned HTTP {response.status}: {text[:200]}"
                    if response.status in _RETRYABLE_HTTP_STATUS:
                        raise RuntimeError(msg)
                    raise _TedHttpStatusError(msg)
                text = await response.text()
                if cache_path is not None:
                    tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
                    tmp_path.write_text(text)
                    tmp_path.replace(cache_path)
                return text
        except _TedHttpStatusError:
            raise  # definitive server answer; retrying will not help
        except Exception as exc:
            last_error = exc
            if attempt < config.retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 10))
    msg = f"Failed to download {ted_id} after {config.retries} attempts"
    if last_error is not None:
        msg = f"{msg}: {last_error}"
    raise RuntimeError(msg)


def assemble_ted_domain_pdbs(pdb_a: str, pdb_b: str) -> str:
    """Assemble two TED-domain PDB strings into a two-chain dimer PDB.

    Args:
        pdb_a: Source PDB text for chain A.
        pdb_b: Source PDB text for chain B.

    Returns:
        Combined PDB text with chains A/B and per-chain residue numbering from 1.
    """
    lines: list[str] = []
    serial = 1
    chain_a, serial = _normalize_pdb_chain(pdb_a, "A", serial)
    chain_b, serial = _normalize_pdb_chain(pdb_b, "B", serial)
    lines.extend(chain_a)
    lines.append("TER\n")
    lines.extend(chain_b)
    lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


def _normalize_pdb_chain(
    pdb_text: str,
    target_chain: str,
    start_serial: int,
) -> tuple[list[str], int]:
    """Normalize one PDB text block to a target chain and residue numbering."""
    residue_map: dict[tuple[str, str], int] = {}
    lines: list[str] = []
    serial = start_serial
    next_residue = 1

    for raw_line in pdb_text.splitlines():
        record = raw_line[:6]
        if record not in {"ATOM  ", "HETATM"}:
            continue
        padded = raw_line.rstrip("\n").ljust(80)
        residue_key = (padded[22:26], padded[26])
        if residue_key not in residue_map:
            residue_map[residue_key] = next_residue
            next_residue += 1
        resseq = residue_map[residue_key]
        line = f"{padded[:6]}{serial:5d}{padded[11:21]}{target_chain}{resseq:4d} {padded[27:]}"
        lines.append(line.rstrip() + "\n")
        serial += 1

    if not lines:
        msg = "TED-domain PDB did not contain any ATOM/HETATM records"
        raise ValueError(msg)
    return lines, serial


def _looks_like_complete_pdb(path: Path) -> bool:
    """Return True when a PDB file exists and ends with an END record."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        tail = path.read_text(errors="ignore")[-256:]
    except OSError:
        return False
    return "END" in tail
