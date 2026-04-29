"""Teddymer synthetic dimer data acquisition pipeline.

Downloads metadata + a bundled FoldSeek structure database, filters by quality
metrics, then extracts each dimer's two TED-domain structures directly from the
FoldSeek DB (CA coordinates + PULCHRA backbone reconstruction via
``foldseek convert2pdb``) and assembles them as chains A/B. No round trip
through AlphaFold DB.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

TEDDYMER_METADATA_URL = "https://teddymer.steineggerlab.workers.dev/foldseek/teddymer.tar.gz"

# Quality filter defaults (from original paper)
DEFAULT_MIN_INTERFACE_PLDDT = 70.0
DEFAULT_MAX_INTERFACE_PAE = 10.0
DEFAULT_MIN_INTERFACE_LENGTH = 10


# ---------------------------------------------------------------------------
# Step 1: Download metadata
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest`` while showing a rich progress bar.

    Writes to ``dest.with_suffix(dest.suffix + ".part")`` and renames on success
    so a partial file is never mistaken for a complete one.
    """
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    # Cloudflare in front of teddymer.steineggerlab.workers.dev rejects the
    # default Python-urllib User-Agent with HTTP 403, so spoof a browser UA.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - trusted https URL
        total = int(resp.headers.get("Content-Length", 0)) or None
        columns = (
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*columns) as progress, tmp.open("wb") as f:
            task_id = progress.add_task(dest.name, total=total)
            while chunk := resp.read(chunk_size):
                f.write(chunk)
                progress.advance(task_id, len(chunk))
    tmp.rename(dest)


def download_teddymer_metadata(output_dir: str | Path) -> Path:
    """Download and extract teddymer metadata tarball.

    Downloads cluster assignments and quality metrics from the Steinegger lab
    server, extracting ``cluster.tsv`` and ``nonsingletonrep_metadata.tsv``.

    Args:
        output_dir: Directory to write extracted files.

    Returns:
        Path to the metadata directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "teddymer.tar.gz"
    if not tar_path.exists():
        logger.info("Downloading teddymer metadata from %s", TEDDYMER_METADATA_URL)
        _download_with_progress(TEDDYMER_METADATA_URL, tar_path)

    logger.info("Extracting teddymer metadata to %s", output_dir)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=output_dir, filter="data")

    return output_dir


# ---------------------------------------------------------------------------
# Step 2: Filter clusters
# ---------------------------------------------------------------------------


# AlphaFold-style TED ID, e.g. "AF-Q9Y6K1-F1-model_v4_TED01".
_TED_ID_UNIPROT_RE = re.compile(r"AF-([A-Z0-9]+)-F\d+")


def _derive_chopping_columns(
    df: pd.DataFrame,
    metadata_dir: Path,
) -> pd.DataFrame:
    """Add ``uniprot_id``/``domain1_chopping``/``domain2_chopping`` columns.

    The teddymer release does not commit to a single column schema, so this
    function detects and normalizes a few common shapes:

    1. The frame already has ``domain1_chopping`` and ``domain2_chopping``
       (pass-through).
    2. The frame has ``domain{1,2}_start`` / ``domain{1,2}_end`` columns —
       build ``"<start>-<end>"`` strings.
    3. The frame has ``ted_id1`` / ``ted_id2`` columns — join against a TED
       domain boundary table found under ``metadata_dir`` (any ``*.tsv`` file
       containing both ``ted_id`` and ``chopping`` columns).
    4. Native teddymer release shape: ``dimerindex`` + ``uniprotid`` +
       ``domainpair`` (e.g. ``"TED01:TED02"``). TED IDs are composed as
       ``"{dimerindex}DI_{uniprotid}_v4_{TEDxx}"`` and chopping is parsed
       from the Foldseek ``*_h`` header file shipped alongside the metadata.

    Also derives ``uniprot_id`` from a TED ID column when missing.

    Returns the input DataFrame with the contract columns populated. Rows whose
    chopping cannot be resolved are dropped with a warning.
    """
    cols = set(df.columns)

    # Already in target form — nothing to do.
    if {"domain1_chopping", "domain2_chopping"}.issubset(cols):
        return df

    # Schema B: explicit start/end columns.
    if {"domain1_start", "domain1_end", "domain2_start", "domain2_end"}.issubset(cols):
        df = df.copy()
        df["domain1_chopping"] = (
            df["domain1_start"].astype(int).astype(str)
            + "-"
            + df["domain1_end"].astype(int).astype(str)
        )
        df["domain2_chopping"] = (
            df["domain2_start"].astype(int).astype(str)
            + "-"
            + df["domain2_end"].astype(int).astype(str)
        )
        return df

    # Schema D: native teddymer release.
    if {"dimerindex", "uniprotid", "domainpair"}.issubset(cols):
        df = df.copy()
        pair = df["domainpair"].astype(str).str.split(":", n=1, expand=True)
        prefix = df["dimerindex"].astype(str) + "DI_" + df["uniprotid"].astype(str) + "_v4_"
        df["ted_id1"] = prefix + pair[0]
        df["ted_id2"] = prefix + pair[1]
        cols = set(df.columns)  # ted_id{1,2} now present, fall through to schema C

    # Schema C: TED IDs + side-table lookup.
    if {"ted_id1", "ted_id2"}.issubset(cols):
        needed = set(df["ted_id1"]).union(df["ted_id2"])
        ted_lookup = _load_ted_domain_table(metadata_dir, needed)
        if ted_lookup is None:
            msg = (
                f"Found ted_id1/ted_id2 columns but no TED domain boundary table under "
                f"{metadata_dir}. Expected a TSV with 'ted_id'+'chopping' or a Foldseek "
                f"'*_h' header file."
            )
            raise FileNotFoundError(msg)
        df["domain1_chopping"] = df["ted_id1"].map(ted_lookup)
        df["domain2_chopping"] = df["ted_id2"].map(ted_lookup)
        if "uniprot_id" not in df.columns:
            df["uniprot_id"] = df["ted_id1"].astype(str).str.extract(_TED_ID_UNIPROT_RE.pattern)[0]
        # Drop rows with unresolved choppings.
        n_before = len(df)
        df = df.dropna(subset=["domain1_chopping", "domain2_chopping"]).copy()
        if len(df) < n_before:
            logger.warning("Dropped %d rows lacking TED chopping entries", n_before - len(df))
        return df

    msg = (
        "Cannot derive domain1_chopping/domain2_chopping from teddymer metadata. "
        f"Columns present: {sorted(cols)}. Provide a metadata schema with one of: "
        "(a) 'domain1_chopping'+'domain2_chopping', "
        "(b) 'domain{1,2}_start'+'domain{1,2}_end', "
        "(c) 'ted_id1'+'ted_id2' with a TED domain table in the metadata directory, or "
        "(d) 'dimerindex'+'uniprotid'+'domainpair' with a Foldseek '*_h' header file."
    )
    raise ValueError(msg)


def _load_ted_domain_table(
    metadata_dir: Path,
    needed: set[str] | None = None,
) -> dict[str, str] | None:
    """Locate a TED domain boundary table and return ``{ted_id: chopping}``.

    Searches ``metadata_dir`` recursively for either:

    * A ``*.tsv``/``*.csv`` containing both ``ted_id`` and ``chopping`` columns,
      or
    * A Foldseek MMseqs2-style header file (``*_h`` next to a matching
      ``*_h.dbtype``) whose null-terminated entries look like
      ``"<ted_id>\\t<CATH>_RES<start>-<end>[_<start>-<end>...]"``.

    When ``needed`` is provided, the Foldseek path filters entries to only that
    set during streaming — required for the ~1.4 GB teddymer header file.

    Returns ``None`` if no source is found.
    """
    for path in metadata_dir.rglob("*.tsv"):
        try:
            head = pd.read_csv(path, sep="\t", nrows=0)
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
        cols = {c.strip().lower() for c in head.columns}
        if {"ted_id", "chopping"}.issubset(cols):
            full = pd.read_csv(path, sep="\t")
            full.columns = full.columns.str.strip().str.lower()
            return dict(zip(full["ted_id"].astype(str), full["chopping"].astype(str), strict=False))

    return _load_ted_domain_table_from_foldseek(metadata_dir, needed)


_FOLDSEEK_RES_RE = re.compile(rb"RES([\d_-]+)")


def _load_ted_domain_table_from_foldseek(
    metadata_dir: Path,
    needed: set[str] | None,
) -> dict[str, str] | None:
    """Stream a Foldseek ``*_h`` header file into ``{ted_id: chopping}``.

    Header entries are null-terminated ``"<id>\\t<CATH>_RES<ranges>"`` records.
    Ranges use ``_`` between segments (e.g. ``"15-30_37-122"``); we rewrite to
    the comma-separated form expected by :func:`_parse_chopping`.
    """
    candidates = [
        p
        for p in metadata_dir.rglob("*_h")
        if p.is_file() and p.suffix == "" and p.with_suffix(".dbtype").exists()
    ]
    if not candidates:
        return None

    # Prefer the smallest matching file (rep-only DB beats full DB if both exist).
    candidates.sort(key=lambda p: p.stat().st_size)
    path = candidates[0]
    logger.info("Parsing TED chopping from Foldseek header file %s", path)

    lookup: dict[str, str] = {}
    with path.open("rb") as f:
        data = f.read()
    for entry in data.split(b"\0"):
        tab = entry.find(b"\t")
        if tab < 0:
            continue
        ted_id = entry[:tab].decode("ascii", errors="ignore").strip()
        if not ted_id or (needed is not None and ted_id not in needed):
            continue
        m = _FOLDSEEK_RES_RE.search(entry, tab)
        if not m:
            continue
        lookup[ted_id] = m.group(1).decode("ascii").replace("_", ",")
    return lookup or None


def filter_teddymer_clusters(
    metadata_dir: str | Path,
    output_path: str | Path,
    *,
    min_interface_plddt: float = DEFAULT_MIN_INTERFACE_PLDDT,
    max_interface_pae: float = DEFAULT_MAX_INTERFACE_PAE,
    min_interface_length: int = DEFAULT_MIN_INTERFACE_LENGTH,
    require_chopping: bool = True,
) -> pd.DataFrame:
    """Filter teddymer cluster representatives by quality metrics.

    Reads ``nonsingletonrep_metadata.tsv``, applies quality filters, derives the
    ``domain1_chopping`` / ``domain2_chopping`` columns required by
    :func:`chop_and_assemble_dimers`, and writes the filtered manifest.

    Args:
        metadata_dir: Directory containing extracted teddymer metadata.
        output_path: Path to write filtered manifest TSV.
        min_interface_plddt: Minimum interface pLDDT score.
        max_interface_pae: Maximum average interface PAE.
        min_interface_length: Minimum number of interface residues.
        require_chopping: If True (default), raise when chopping columns
            cannot be derived. Set False to keep this function usable as a
            quality filter on metadata that lacks domain information.

    Returns:
        Filtered manifest as a DataFrame.
    """
    metadata_dir = Path(metadata_dir)
    output_path = Path(output_path)

    metadata_path = metadata_dir / "nonsingletonrep_metadata.tsv"
    if not metadata_path.exists():
        # Try nested directory from tar extraction
        candidates = list(metadata_dir.rglob("nonsingletonrep_metadata.tsv"))
        if not candidates:
            msg = f"nonsingletonrep_metadata.tsv not found in {metadata_dir}"
            raise FileNotFoundError(msg)
        metadata_path = candidates[0]

    logger.info("Reading teddymer metadata from %s", metadata_path)
    df = pd.read_csv(metadata_path, sep="\t")

    # Normalize column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Apply quality filters
    n_before = len(df)
    mask = (
        (df["avgintplddt"] > min_interface_plddt)
        & (df["avgintpae"] < max_interface_pae)
        & (df["interfacelength"] > min_interface_length)
    )
    df = df[mask].copy()
    logger.info(
        "Filtered teddymer clusters: %d → %d (%.1f%% retained)",
        n_before,
        len(df),
        100.0 * len(df) / max(n_before, 1),
    )

    # Derive chopping columns required by chop_and_assemble_dimers.
    try:
        df = _derive_chopping_columns(df, metadata_dir)
    except (ValueError, FileNotFoundError):
        if require_chopping:
            raise
        logger.warning(
            "Could not derive chopping columns from %s; manifest will lack "
            "domain{1,2}_chopping and chop_and_assemble_dimers will fail.",
            metadata_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote filtered manifest to %s", output_path)

    return df


# ---------------------------------------------------------------------------
# Step 3: Extract dimers from the bundled FoldSeek structure DB
# ---------------------------------------------------------------------------


def _parse_chopping(chopping: str) -> list[tuple[int, int]]:
    """Parse a CATH-style domain chopping string into residue ranges.

    Handles formats like ``"10-50"`` or ``"10-50,80-120"`` (discontinuous).

    Args:
        chopping: Chopping string, e.g. ``"10-50,80-120"``.

    Returns:
        List of (start, end) residue ranges (inclusive).
    """
    ranges: list[tuple[int, int]] = []
    for segment in chopping.split(","):
        segment = segment.strip()
        if not segment:
            continue
        # Strip optional chain prefix (e.g., "A:10-50" → "10-50")
        if ":" in segment:
            segment = segment.split(":", 1)[1]
        parts = segment.split("-")
        ranges.append((int(parts[0]), int(parts[1])))
    return ranges


_FOLDSEEK_DB_SUFFIXES = (
    "",
    "_ca",
    "_h",
    "_ss",
    ".dbtype",
    ".index",
    ".lookup",
    ".source",
)


def _check_foldseek_db(db: Path) -> None:
    """Verify all sibling files of a FoldSeek structure DB are present.

    Raises ``FileNotFoundError`` listing every missing path so the user can fix
    the extraction in one go.
    """
    missing = [
        str(db) + suffix
        for suffix in _FOLDSEEK_DB_SUFFIXES
        if not (db.with_name(db.name + suffix).exists())
    ]
    if missing:
        msg = "FoldSeek DB is incomplete; missing files:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)


_FOLDSEEK_SIBLING_DBS = ("_ca", "_h", "_ss")


def _ensure_sibling_lookups(db: Path) -> None:
    """Create per-sibling ``.lookup``/``.source`` symlinks if missing.

    The teddymer release ships one ``.lookup`` and ``.source`` on the main DB
    only, but ``foldseek createsubdb`` insists on per-sibling copies. The
    sibling DBs share the main DB's id mapping, so a symlink is the canonical
    fix. Idempotent: existing files (or symlinks) are left untouched.
    """
    main_lookup = db.with_name(db.name + ".lookup")
    main_source = db.with_name(db.name + ".source")
    for sibling in _FOLDSEEK_SIBLING_DBS:
        for ext, target in ((".lookup", main_lookup), (".source", main_source)):
            link = db.with_name(db.name + sibling + ext)
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(target.name)


def _run_foldseek(args: list[str], *, foldseek_binary: str = "foldseek") -> None:
    """Run a foldseek subcommand, surfacing stderr through the logger."""
    cmd = [foldseek_binary, *args]
    logger.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = f"foldseek failed (exit {proc.returncode}): {' '.join(cmd)}\nstderr:\n{proc.stderr}"
        raise RuntimeError(msg)
    if proc.stderr:
        logger.debug("foldseek stderr: %s", proc.stderr.strip())


def _assemble_one(ted1_pdb: Path, ted2_pdb: Path, output_path: Path) -> bool:
    """Combine two single-chain PDBs into a chain-A/B dimer with 1-based numbering.

    PULCHRA emits one chain per file with arbitrary chain IDs and original
    residue numbering. We rename the chains to ``A`` and ``B`` and renumber
    residues from 1 in each chain. The two-pass renumber avoids ID collisions
    when the original numbering overlaps with the target range.

    Args:
        ted1_pdb: Path to the chain-A source PDB (e.g. ``<ted_id1>.pdb``).
        ted2_pdb: Path to the chain-B source PDB.
        output_path: Path to write the assembled dimer PDB.

    Returns:
        True on success; False if either source file is missing or unparseable.
    """
    from Bio.PDB import PDBIO, PDBParser  # type: ignore[attr-defined]
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    if output_path.exists():
        return True
    if not ted1_pdb.exists() or not ted2_pdb.exists():
        return False

    try:
        parser = PDBParser(QUIET=True)  # type: ignore[no-untyped-call]

        def _normalize(src: Path, target_chain: str):  # type: ignore[no-untyped-def]
            structure = parser.get_structure(target_chain, str(src))  # type: ignore[no-untyped-call]
            chain = next(structure[0].get_chains())
            chain.id = target_chain
            residues = list(chain)
            # Two-pass renumber: stash to a sentinel hetflag first to avoid
            # collisions when the source numbering overlaps 1..N.
            for i, residue in enumerate(residues, start=1):
                residue.id = ("X", i, " ")
            for residue in residues:
                _, idx, _ = residue.id
                residue.id = (" ", idx, " ")
            return chain

        chain_a = _normalize(ted1_pdb, "A")
        chain_b = _normalize(ted2_pdb, "B")

        out_structure = Structure("dimer")  # type: ignore[no-untyped-call]
        model = Model(0)  # type: ignore[no-untyped-call]
        model.add(chain_a)
        model.add(chain_b)
        out_structure.add(model)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()  # type: ignore[no-untyped-call]
        io.set_structure(out_structure)  # type: ignore[no-untyped-call]
        io.save(str(output_path))  # type: ignore[no-untyped-call]
        return True
    except Exception:
        logger.warning("Failed to assemble dimer from %s + %s", ted1_pdb, ted2_pdb, exc_info=True)
        return False


def _assemble_one_with_meta(
    ted1_pdb: Path,
    ted2_pdb: Path,
    output_path: Path,
    source_id: str,
    interface_residues: int,
) -> dict[str, str | int] | None:
    """Worker for the assembly pool: assemble + return manifest record on success."""
    if not _assemble_one(ted1_pdb, ted2_pdb, output_path):
        return None
    return {
        "structure_path": str(output_path),
        "chain_A": "A",
        "chain_B": "B",
        "source": "teddymer",
        "source_id": source_id,
        "split_group": source_id,
        "interface_residues": interface_residues,
    }


def _select_source_id(row: pd.Series) -> str:
    """Pick the most stable identifier available on a manifest row."""
    for column in ("cluster_rep", "cluster_id", "dimerindex", "uniprotid", "uniprot_id"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return str(row.name)


def _output_path_for_row(row: pd.Series, output_dir: Path) -> Path:
    """Stable per-row dimer path. Prefers ``dimerindex`` when present."""
    if "dimerindex" in row and pd.notna(row["dimerindex"]):
        stem = str(row["dimerindex"])
    else:
        stem = str(row.name)
    return output_dir / f"{stem}.pdb"


def extract_and_assemble_dimers(
    manifest_path: str | Path,
    foldseek_db: str | Path,
    output_dir: str | Path,
    *,
    scratch_dir: str | Path | None = None,
    chunk_size: int = 50_000,
    workers: int = 16,
    foldseek_threads: int | None = None,
    foldseek_binary: str = "foldseek",
    keep_scratch: bool = False,
) -> int:
    """Extract TED-domain structures from a FoldSeek DB and assemble dimers.

    Replaces the previous AFDB-download + ``pdb-tools``-chop pipeline. For each
    row of the filtered manifest, fetches the two TED-domain PDBs from the
    bundled FoldSeek DB (``foldseek convert2pdb`` reconstructs N/C/O via
    PULCHRA, ~0.5 Å RMSD vs. true AFDB backbone) and concatenates them as
    chains A/B with 1-based residue numbering.

    Processed in chunks of ``chunk_size`` rows: per chunk, runs
    ``foldseek createsubdb`` + ``foldseek convert2pdb`` on the unique TED IDs,
    assembles dimers in parallel, then deletes the per-chunk scratch.

    Args:
        manifest_path: Path to filtered teddymer manifest TSV. Required columns:
            ``ted_id1``, ``ted_id2``. Optional but used: ``dimerindex``,
            ``cluster_rep``, ``cluster_id``, ``uniprotid``, ``uniprot_id``,
            ``interfacelength``.
        foldseek_db: Path to the FoldSeek structure DB *prefix* (no suffix);
            sibling files ``<db>_ca``, ``<db>_h``, ``<db>.lookup``, etc. must
            exist.
        output_dir: Directory to write assembled dimer PDBs and ``manifest.tsv``.
        scratch_dir: Directory for per-chunk subset DBs and PDBs. Defaults to
            ``<output_dir>/.scratch``. Must be on the same or larger filesystem
            (~3 GB per 50K-row chunk).
        chunk_size: Number of manifest rows per FoldSeek extraction batch.
        workers: Process pool size for the BioPython assembly step.
        foldseek_threads: ``--threads`` value forwarded to ``foldseek convert2pdb``;
            ``None`` lets foldseek use all available cores.
        foldseek_binary: Name or path of the foldseek executable.
        keep_scratch: If True, retain per-chunk scratch dirs for debugging.

    Returns:
        Number of successfully assembled dimers.

    Raises:
        RuntimeError: if the foldseek binary is not on PATH.
        FileNotFoundError: if the FoldSeek DB has missing sibling files or the
            manifest lacks ``ted_id1``/``ted_id2`` columns.
    """
    if shutil.which(foldseek_binary) is None:
        msg = (
            f"foldseek binary {foldseek_binary!r} not found on PATH. Install it "
            "with `mamba install -c conda-forge -c bioconda foldseek` or download "
            "the static binary from https://github.com/steineggerlab/foldseek/releases."
        )
        raise RuntimeError(msg)

    foldseek_db = Path(foldseek_db)
    _check_foldseek_db(foldseek_db)
    _ensure_sibling_lookups(foldseek_db)

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir) if scratch_dir is not None else output_dir / ".scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")
    if not {"ted_id1", "ted_id2"}.issubset(df.columns):
        msg = (
            f"Filtered manifest at {manifest_path} is missing required "
            f"columns 'ted_id1' and 'ted_id2'. Re-run filter_teddymer_clusters."
        )
        raise FileNotFoundError(msg)

    logger.info("Assembling up to %d dimers from FoldSeek DB %s", len(df), foldseek_db)

    success = 0
    missing_entries = 0
    manifest_records: list[dict[str, str | int]] = []

    with Progress() as progress:
        task_id = progress.add_task("Assembling dimers", total=len(df))

        for chunk_idx, start in enumerate(range(0, len(df), chunk_size)):
            chunk = df.iloc[start : start + chunk_size]

            # Resume: skip rows whose output already exists.
            chunk_remaining = chunk[
                ~chunk.apply(lambda r: _output_path_for_row(r, output_dir).exists(), axis=1)
            ]
            already_done = len(chunk) - len(chunk_remaining)
            if already_done:
                progress.advance(task_id, already_done)
                success += already_done
            if chunk_remaining.empty:
                continue

            ted_ids = pd.unique(
                pd.concat([chunk_remaining["ted_id1"], chunk_remaining["ted_id2"]]).astype(str)
            )
            ids_path = scratch_dir / f"chunk_{chunk_idx:04d}_ids.txt"
            ids_path.write_text("\n".join(ted_ids) + "\n")

            subset_db = scratch_dir / f"chunk_{chunk_idx:04d}_db"
            pdb_dir = scratch_dir / f"chunk_{chunk_idx:04d}_pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)

            _run_foldseek(
                [
                    "createsubdb",
                    "--id-mode",
                    "1",
                    "--subdb-mode",
                    "1",
                    str(ids_path),
                    str(foldseek_db),
                    str(subset_db),
                ],
                foldseek_binary=foldseek_binary,
            )
            convert_args = [
                "convert2pdb",
                "--pdb-output-mode",
                "1",
                str(subset_db),
                str(pdb_dir),
            ]
            if foldseek_threads is not None:
                convert_args.extend(["--threads", str(foldseek_threads)])
            _run_foldseek(convert_args, foldseek_binary=foldseek_binary)

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = []
                for _, row in chunk_remaining.iterrows():
                    ted1 = pdb_dir / f"{row['ted_id1']}.pdb"
                    ted2 = pdb_dir / f"{row['ted_id2']}.pdb"
                    output_path = _output_path_for_row(row, output_dir)
                    source_id = _select_source_id(row)
                    interface_residues = int(row.get("interfacelength", 0) or 0)
                    futures.append(
                        pool.submit(
                            _assemble_one_with_meta,
                            ted1,
                            ted2,
                            output_path,
                            source_id,
                            interface_residues,
                        )
                    )
                for fut in futures:
                    record = fut.result()
                    if record is None:
                        missing_entries += 1
                    else:
                        success += 1
                        manifest_records.append(record)
                    progress.advance(task_id)

            if not keep_scratch:
                ids_path.unlink(missing_ok=True)
                shutil.rmtree(pdb_dir, ignore_errors=True)
                for suffix in _FOLDSEEK_DB_SUFFIXES:
                    sibling = subset_db.with_name(subset_db.name + suffix)
                    if sibling.exists() or sibling.is_symlink():
                        sibling.unlink()

    if manifest_records:
        manifest = pd.DataFrame(manifest_records)
        out_manifest = output_dir / "manifest.tsv"
        manifest.to_csv(out_manifest, sep="\t", index=False)
        logger.info("Wrote teddymer training manifest to %s", out_manifest)

    if missing_entries:
        logger.warning(
            "%d dimers skipped because a TED entry was absent from the FoldSeek DB",
            missing_entries,
        )
    logger.info("Assembled %d / %d dimers", success, len(df))

    if not keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return success
