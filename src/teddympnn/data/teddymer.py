"""Teddymer synthetic dimer data acquisition pipeline.

Downloads and preprocesses teddymer synthetic dimers constructed from AFDB
domain pairs for PPI fine-tuning. Each pipeline step is a standalone function
that can be invoked from the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rich.progress import Progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

TEDDYMER_METADATA_URL = "https://teddymer.steineggerlab.workers.dev/foldseek/teddymer.tar"
AFDB_STRUCTURE_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

# Quality filter defaults (from original paper)
DEFAULT_MIN_INTERFACE_PLDDT = 70.0
DEFAULT_MAX_INTERFACE_PAE = 10.0
DEFAULT_MIN_INTERFACE_LENGTH = 10


# ---------------------------------------------------------------------------
# Step 1: Download metadata
# ---------------------------------------------------------------------------


def download_teddymer_metadata(output_dir: str | Path) -> Path:
    """Download and extract teddymer metadata tarball.

    Downloads cluster assignments and quality metrics from the Steinegger lab
    server, extracting ``cluster.tsv`` and ``nonsingletonrep_metadata.tsv``.

    Args:
        output_dir: Directory to write extracted files.

    Returns:
        Path to the metadata directory.
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "teddymer.tar"
    if not tar_path.exists():
        logger.info("Downloading teddymer metadata from %s", TEDDYMER_METADATA_URL)
        urllib.request.urlretrieve(TEDDYMER_METADATA_URL, tar_path)

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

    # Schema C: TED IDs + side-table lookup.
    if {"ted_id1", "ted_id2"}.issubset(cols):
        ted_lookup = _load_ted_domain_table(metadata_dir)
        if ted_lookup is None:
            msg = (
                f"Found ted_id1/ted_id2 columns but no TED domain boundary table under "
                f"{metadata_dir}. Expected a TSV containing 'ted_id' and 'chopping'."
            )
            raise FileNotFoundError(msg)
        df = df.copy()
        df["domain1_chopping"] = df["ted_id1"].map(ted_lookup)
        df["domain2_chopping"] = df["ted_id2"].map(ted_lookup)
        if "uniprot_id" not in cols:
            df["uniprot_id"] = (
                df["ted_id1"].astype(str).str.extract(_TED_ID_UNIPROT_RE.pattern)[0]
            )
        # Drop rows with unresolved choppings.
        n_before = len(df)
        df = df.dropna(subset=["domain1_chopping", "domain2_chopping"]).copy()
        if len(df) < n_before:
            logger.warning(
                "Dropped %d rows lacking TED chopping entries", n_before - len(df)
            )
        return df

    msg = (
        "Cannot derive domain1_chopping/domain2_chopping from teddymer metadata. "
        f"Columns present: {sorted(cols)}. Provide a metadata schema with one of: "
        "(a) 'domain1_chopping'+'domain2_chopping', "
        "(b) 'domain{1,2}_start'+'domain{1,2}_end', or "
        "(c) 'ted_id1'+'ted_id2' with a TED domain table in the metadata directory."
    )
    raise ValueError(msg)


def _load_ted_domain_table(metadata_dir: Path) -> dict[str, str] | None:
    """Locate a TED domain boundary table and return ``{ted_id: chopping}``.

    Searches ``metadata_dir`` recursively for any ``*.tsv``/``*.csv`` containing
    both ``ted_id`` and ``chopping`` columns. Returns ``None`` if nothing is
    found.
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
    return None


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
        (df["interfaceplddt"] > min_interface_plddt)
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
# Step 3: Download AFDB structures
# ---------------------------------------------------------------------------


async def _download_one(
    session: object,  # aiohttp.ClientSession
    uniprot_id: str,
    output_dir: Path,
) -> bool:
    """Download a single AFDB structure."""
    import aiohttp

    assert isinstance(session, aiohttp.ClientSession)
    output_path = output_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
    if output_path.exists():
        return True

    url = AFDB_STRUCTURE_URL.format(uniprot_id=uniprot_id)
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning("Failed to download %s: HTTP %d", uniprot_id, resp.status)
                return False
            content = await resp.read()
            output_path.write_bytes(content)
            return True
    except Exception:
        logger.warning("Error downloading %s", uniprot_id, exc_info=True)
        return False


async def _download_batch(
    uniprot_ids: list[str],
    output_dir: Path,
    workers: int,
) -> int:
    """Download AFDB structures concurrently."""
    import aiohttp

    connector = aiohttp.TCPConnector(limit=workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_download_one(session, uid, output_dir) for uid in uniprot_ids]
        results = await asyncio.gather(*tasks)
    return sum(results)


def download_afdb_structures(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    uniprot_column: str = "uniprot_id",
    workers: int = 50,
) -> int:
    """Download full-chain PDB structures from AFDB.

    Reads UniProt IDs from the filtered manifest and downloads the
    corresponding AlphaFold DB structures concurrently.

    Args:
        manifest_path: Path to filtered teddymer manifest TSV.
        output_dir: Directory to save downloaded PDB files.
        uniprot_column: Column name containing UniProt IDs.
        workers: Number of concurrent download workers.

    Returns:
        Number of successfully downloaded structures.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")
    uniprot_ids = df[uniprot_column].unique().tolist()
    logger.info("Downloading %d unique AFDB structures", len(uniprot_ids))

    downloaded = asyncio.run(_download_batch(uniprot_ids, output_dir, workers))
    logger.info("Downloaded %d / %d structures", downloaded, len(uniprot_ids))
    return downloaded


# ---------------------------------------------------------------------------
# Step 4: Chop domains and assemble dimers
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


def _chop_and_assemble_one(
    afdb_path: Path,
    domain1_chopping: str,
    domain2_chopping: str,
    output_path: Path,
) -> bool:
    """Chop two domains from a full-chain PDB and assemble a dimer.

    Uses pdb-tools for residue selection, chain relabeling, and residue
    renumbering.

    Args:
        afdb_path: Path to the full-chain AFDB PDB file.
        domain1_chopping: Chopping string for domain 1 (chain A).
        domain2_chopping: Chopping string for domain 2 (chain B).
        output_path: Path to write the assembled dimer PDB.

    Returns:
        True if successful.
    """
    from pdbtools import pdb_chain, pdb_reres, pdb_selres

    if output_path.exists():
        return True

    try:
        with open(afdb_path) as f:
            lines = f.readlines()

        # Chop domain 1
        ranges1 = _parse_chopping(domain1_chopping)
        range_str1 = ",".join(f"{s}:{e}" for s, e in ranges1)
        d1_lines = list(pdb_selres.run(lines, range_str1))
        d1_lines = list(pdb_chain.run(d1_lines, "A"))
        d1_lines = list(pdb_reres.run(d1_lines, 1))

        # Chop domain 2
        ranges2 = _parse_chopping(domain2_chopping)
        range_str2 = ",".join(f"{s}:{e}" for s, e in ranges2)
        d2_lines = list(pdb_selres.run(lines, range_str2))
        d2_lines = list(pdb_chain.run(d2_lines, "B"))
        d2_lines = list(pdb_reres.run(d2_lines, 1))

        # Assemble dimer: domain 1 (chain A) + domain 2 (chain B)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for line in d1_lines:
                if line.startswith(("ATOM", "HETATM", "TER")):
                    f.write(line)
            for line in d2_lines:
                if line.startswith(("ATOM", "HETATM", "TER")):
                    f.write(line)
            f.write("END\n")

        return True
    except Exception:
        logger.warning("Failed to assemble dimer from %s", afdb_path, exc_info=True)
        return False


def chop_and_assemble_dimers(
    manifest_path: str | Path,
    afdb_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 50,
) -> int:
    """Chop domains and assemble dimers from AFDB structures.

    Reads the filtered manifest, chops domains from downloaded AFDB PDB files,
    relabels chains as A and B, and writes assembled dimer PDBs.

    Args:
        manifest_path: Path to filtered teddymer manifest TSV.
        afdb_dir: Directory containing downloaded AFDB PDB files.
        output_dir: Directory to write assembled dimer PDB files.
        workers: Number of parallel worker processes.

    Returns:
        Number of successfully assembled dimers.
    """
    manifest_path = Path(manifest_path)
    afdb_dir = Path(afdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")
    logger.info("Assembling %d dimers", len(df))

    required = {"uniprot_id", "domain1_chopping", "domain2_chopping"}
    missing = required.difference(df.columns)
    if missing:
        msg = f"Teddymer manifest missing columns required for assembly: {sorted(missing)}"
        raise ValueError(msg)

    tasks: list[tuple[Path, str, str, Path, pd.Series]] = []
    for _, row in df.iterrows():
        uniprot_id = row["uniprot_id"]
        afdb_path = afdb_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
        if not afdb_path.exists():
            continue
        output_path = output_dir / f"{row.name}.pdb"
        chop1 = str(row["domain1_chopping"])
        chop2 = str(row["domain2_chopping"])
        tasks.append((afdb_path, chop1, chop2, output_path, row))

    success = 0
    manifest_records: list[dict[str, str | int]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            (
                pool.submit(_chop_and_assemble_one, afdb_path, chop1, chop2, output_path),
                output_path,
                row,
            )
            for afdb_path, chop1, chop2, output_path, row in tasks
        ]
        with Progress() as progress:
            task_id = progress.add_task("Assembling dimers", total=len(futures))
            for future, output_path, row in futures:
                if future.result():
                    success += 1
                    source_id = str(
                        row.get("cluster_rep", row.get("cluster_id", row.get("uniprot_id", row.name)))
                    )
                    manifest_records.append(
                        {
                            "structure_path": str(output_path),
                            "chain_A": "A",
                            "chain_B": "B",
                            "source": "teddymer",
                            "source_id": source_id,
                            "split_group": source_id,
                            "interface_residues": int(row.get("interfacelength", 0)),
                        }
                    )
                progress.advance(task_id)

    if manifest_records:
        manifest = pd.DataFrame(manifest_records)
        manifest_path = output_dir / "manifest.tsv"
        manifest.to_csv(manifest_path, sep="\t", index=False)
        logger.info("Wrote teddymer training manifest to %s", manifest_path)

    logger.info("Assembled %d / %d dimers", success, len(tasks))
    return success
