"""NVIDIA/EMBL-EBI predicted complexes data acquisition pipeline.

Downloads and filters high-confidence AlphaFold-Multimer predicted complexes
from the NVIDIA/EMBL-EBI dataset (~1.8M structures from ~31M predictions).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rich.progress import Progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs and defaults
# ---------------------------------------------------------------------------

METADATA_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/alphafold/predicted_complexes/"
    "model_entity_metadata_mapping.csv"
)

CHUNK_URL_TEMPLATE = (
    "https://ftp.ebi.ac.uk/pub/databases/alphafold/predicted_complexes/"
    "chunks/chunk_{chunk_id:04d}.tar"
)

# Quality filter defaults (from original paper)
DEFAULT_MIN_IPSAE = 0.6
DEFAULT_MIN_PLDDT = 70.0
DEFAULT_MAX_CLASHES = 10

# Chunk size for reading large CSV
CSV_CHUNKSIZE = 500_000


# ---------------------------------------------------------------------------
# Step 1: Download metadata
# ---------------------------------------------------------------------------


def download_nvidia_metadata(output_dir: str | Path) -> Path:
    """Download the NVIDIA complexes metadata CSV (~4.3 GB).

    Args:
        output_dir: Directory to save the metadata file.

    Returns:
        Path to the downloaded CSV file.
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_entity_metadata_mapping.csv"

    if csv_path.exists():
        logger.info("Metadata already exists at %s", csv_path)
        return csv_path

    logger.info("Downloading NVIDIA complexes metadata (this may take a while)")
    urllib.request.urlretrieve(METADATA_URL, csv_path)
    logger.info("Downloaded metadata to %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Step 2: Filter metadata
# ---------------------------------------------------------------------------


def filter_nvidia_metadata(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    min_ipsae: float = DEFAULT_MIN_IPSAE,
    min_plddt: float = DEFAULT_MIN_PLDDT,
    max_clashes: int = DEFAULT_MAX_CLASHES,
) -> pd.DataFrame:
    """Filter NVIDIA complexes metadata by confidence thresholds.

    Reads the large metadata CSV in chunks, applies quality filters, groups
    passing structures by chunk tarball, and writes a filtered manifest.

    Args:
        csv_path: Path to the metadata CSV.
        output_path: Path to write the filtered manifest.
        min_ipsae: Minimum ipSAE score.
        min_plddt: Minimum average pLDDT.
        max_clashes: Maximum backbone clashes.

    Returns:
        Filtered manifest DataFrame with columns including chunk_id.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    logger.info(
        "Filtering NVIDIA metadata: ipSAE>=%.2f, pLDDT>=%.0f, clashes<=%d",
        min_ipsae,
        min_plddt,
        max_clashes,
    )

    passing_chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CSV_CHUNKSIZE):
        total_rows += len(chunk)
        # Normalize column names
        chunk.columns = chunk.columns.str.strip()

        mask = (
            (chunk["ipSAEmin"] >= min_ipsae)
            & (chunk["pLDDTavg"] >= min_plddt)
            & (chunk["N_clash_backbone"] <= max_clashes)
        )
        passing = chunk[mask]
        if not passing.empty:
            passing_chunks.append(passing)

    if not passing_chunks:
        logger.warning("No structures passed the filters")
        df = pd.DataFrame()
    else:
        df = pd.concat(passing_chunks, ignore_index=True)

    logger.info(
        "Filtered: %d / %d structures passed (%.1f%%)",
        len(df),
        total_rows,
        100.0 * len(df) / max(total_rows, 1),
    )

    # Identify unique chunks needed for download
    if "chunk_id" in df.columns:
        unique_chunks = df["chunk_id"].nunique()
        logger.info("Structures span %d chunk tarballs", unique_chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote filtered manifest to %s", output_path)

    return df


# ---------------------------------------------------------------------------
# Step 3: Download chunk tarballs
# ---------------------------------------------------------------------------


def download_nvidia_chunks(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_column: str = "chunk_id",
    workers: int = 4,
) -> int:
    """Download required chunk tarballs from the NVIDIA complexes FTP.

    Each chunk tarball is ~7.5 GB, so only chunks containing passing
    structures are downloaded.

    Args:
        manifest_path: Path to filtered manifest TSV.
        output_dir: Directory to save chunk tarballs.
        chunk_column: Column name containing chunk IDs.
        workers: Number of concurrent downloads.

    Returns:
        Number of successfully downloaded chunks.
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")
    chunk_ids = sorted(df[chunk_column].unique())
    logger.info("Downloading %d chunk tarballs", len(chunk_ids))

    def _download_one(chunk_id: int) -> bool:
        tar_path = output_dir / f"chunk_{chunk_id:04d}.tar"
        if tar_path.exists():
            return True

        url = CHUNK_URL_TEMPLATE.format(chunk_id=chunk_id)
        try:
            urllib.request.urlretrieve(url, tar_path)
            return True
        except Exception:
            logger.warning("Failed to download chunk %d", chunk_id, exc_info=True)
            return False

    downloaded = 0
    with Progress() as progress:
        task = progress.add_task("Downloading chunks", total=len(chunk_ids))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ok in pool.map(_download_one, chunk_ids):
                if ok:
                    downloaded += 1
                progress.advance(task)

    logger.info("Downloaded %d / %d chunks", downloaded, len(chunk_ids))
    return downloaded


# ---------------------------------------------------------------------------
# Step 4: Extract passing structures
# ---------------------------------------------------------------------------


def extract_nvidia_structures(
    manifest_path: str | Path,
    chunks_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Extract passing structures from downloaded chunk tarballs.

    For each chunk tarball, extracts only the structures listed in the
    filtered manifest. Handles zstd-compressed files within tarballs.

    Args:
        manifest_path: Path to filtered manifest TSV.
        chunks_dir: Directory containing downloaded chunk tarballs.
        output_dir: Directory to write extracted structure files.

    Returns:
        Number of successfully extracted structures.
    """
    import tarfile

    manifest_path = Path(manifest_path)
    chunks_dir = Path(chunks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, sep="\t")

    # Build set of passing structure filenames per chunk
    if "chunk_id" not in df.columns or "filename" not in df.columns:
        logger.warning("Manifest missing chunk_id or filename columns")
        return 0

    chunk_to_filenames: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        cid = int(row["chunk_id"])
        chunk_to_filenames.setdefault(cid, set()).add(str(row["filename"]))

    extracted = 0
    manifest_records: list[dict[str, str | int]] = []
    with Progress() as progress:
        task = progress.add_task("Extracting structures", total=len(chunk_to_filenames))
        for chunk_id, filenames in chunk_to_filenames.items():
            tar_path = chunks_dir / f"chunk_{chunk_id:04d}.tar"
            if not tar_path.exists():
                progress.advance(task)
                continue

            try:
                with tarfile.open(tar_path, "r:*") as tar:
                    for member in tar:
                        basename = Path(member.name).name
                        if basename in filenames:
                            # Handle zstd compression
                            content = tar.extractfile(member)
                            if content is None:
                                continue

                            data = content.read()
                            out_name = basename
                            if basename.endswith(".zst"):
                                try:
                                    import zstandard

                                    dctx = zstandard.ZstdDecompressor()
                                    data = dctx.decompress(data)
                                    out_name = basename[:-4]  # Strip .zst
                                except ImportError:
                                    logger.warning(
                                        "zstandard not installed; skipping %s",
                                        basename,
                                    )
                                    continue

                            out_path = output_dir / out_name
                            out_path.write_bytes(data)
                            extracted += 1
                            matching = df[df["filename"].astype(str).map(Path).map(lambda p: p.name) == basename]
                            row = matching.iloc[0] if not matching.empty else None
                            source_id = (
                                str(row.get("model_id", out_name)) if row is not None else out_name
                            )
                            manifest_records.append(
                                {
                                    "structure_path": str(out_path),
                                    "chain_A": str(row.get("chain_A", "A"))
                                    if row is not None
                                    else "A",
                                    "chain_B": str(row.get("chain_B", "B"))
                                    if row is not None
                                    else "B",
                                    "source": "nvidia",
                                    "source_id": source_id,
                                    "split_group": source_id,
                                    "interface_residues": int(row.get("interface_residues", 0))
                                    if row is not None
                                    else 0,
                                }
                            )
            except Exception:
                logger.warning("Error processing chunk %d", chunk_id, exc_info=True)

            progress.advance(task)

    logger.info("Extracted %d structures", extracted)
    if manifest_records:
        training_manifest = pd.DataFrame(manifest_records)
        manifest_out = output_dir / "manifest.tsv"
        training_manifest.to_csv(manifest_out, sep="\t", index=False)
        logger.info("Wrote NVIDIA training manifest to %s", manifest_out)
    return extracted
