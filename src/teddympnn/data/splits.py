"""Train/validation manifest splitting for PPI datasets.

Creates reproducible train/val splits respecting data source structure:
teddymer splits by cluster, NVIDIA by complex, PDB by structure.
Produces unified manifests that the training pipeline can consume directly.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Required manifest columns for training
MANIFEST_COLUMNS: tuple[str, ...] = (
    "structure_path",
    "chain_A",
    "chain_B",
    "source",
    "source_id",
    "split_group",
    "interface_residues",
)


# ---------------------------------------------------------------------------
# Deterministic hashing for reproducible splits
# ---------------------------------------------------------------------------


def _hash_split(key: str, val_fraction: float, seed: int = 42) -> bool:
    """Deterministically assign a key to the validation set.

    Uses MD5 hash of ``f"{seed}:{key}"`` to get a uniform [0, 1) value,
    then returns True if the value falls below ``val_fraction``.

    Args:
        key: Unique identifier for the split unit (cluster ID, PDB ID, etc.).
        val_fraction: Fraction of keys assigned to validation.
        seed: Random seed baked into the hash for reproducibility.

    Returns:
        True if the key belongs to the validation set.
    """
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return (int(h, 16) % 10_000) / 10_000 < val_fraction


# ---------------------------------------------------------------------------
# Per-source splitting
# ---------------------------------------------------------------------------


def split_teddymer_manifest(
    manifest_path: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    cluster_column: str = "cluster_rep",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split teddymer manifest by cluster representative.

    All members of a cluster go to the same split, preventing data leakage
    from structurally similar domains appearing in both train and val.

    Args:
        manifest_path: Path to filtered teddymer manifest TSV.
        val_fraction: Fraction of clusters assigned to validation.
        seed: Random seed for deterministic splitting.
        cluster_column: Column containing cluster representative IDs.

    Returns:
        Tuple of (train_df, val_df).
    """
    df = pd.read_csv(manifest_path, sep="\t")
    df.columns = df.columns.str.strip().str.lower()

    # Normalize cluster column name
    cluster_col = cluster_column.lower()
    if cluster_col not in df.columns:
        # Fall back: try the first column that contains "cluster" or "rep"
        candidates = [c for c in df.columns if "cluster" in c or "rep" in c]
        if candidates:
            cluster_col = candidates[0]
            logger.info("Using cluster column: %s", cluster_col)
        else:
            msg = f"No cluster column found in {manifest_path}. Columns: {list(df.columns)}"
            raise ValueError(msg)

    # Assign clusters to val or train
    unique_clusters = df[cluster_col].unique()
    val_clusters = {c for c in unique_clusters if _hash_split(str(c), val_fraction, seed)}

    is_val = df[cluster_col].isin(val_clusters)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    logger.info(
        "Teddymer split: %d clusters → %d train (%d rows), %d val (%d rows)",
        len(unique_clusters),
        len(unique_clusters) - len(val_clusters),
        len(train_df),
        len(val_clusters),
        len(val_df),
    )
    return train_df, val_df


def split_nvidia_manifest(
    manifest_path: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    complex_column: str = "model_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split NVIDIA complexes manifest by complex identifier.

    Args:
        manifest_path: Path to filtered NVIDIA manifest TSV.
        val_fraction: Fraction of complexes assigned to validation.
        seed: Random seed for deterministic splitting.
        complex_column: Column containing complex identifiers.

    Returns:
        Tuple of (train_df, val_df).
    """
    df = pd.read_csv(manifest_path, sep="\t")
    df.columns = df.columns.str.strip()

    col = complex_column
    if col not in df.columns:
        candidates = [c for c in df.columns if "model" in c.lower() or "id" in c.lower()]
        if candidates:
            col = candidates[0]
            logger.info("Using complex column: %s", col)
        else:
            msg = f"No complex ID column found in {manifest_path}. Columns: {list(df.columns)}"
            raise ValueError(msg)

    unique_ids = df[col].unique()
    val_ids = {c for c in unique_ids if _hash_split(str(c), val_fraction, seed)}

    is_val = df[col].isin(val_ids)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    logger.info(
        "NVIDIA split: %d complexes → %d train (%d rows), %d val (%d rows)",
        len(unique_ids),
        len(unique_ids) - len(val_ids),
        len(train_df),
        len(val_ids),
        len(val_df),
    )
    return train_df, val_df


def split_pdb_manifest(
    manifest_path: str | Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    structure_column: str = "pdb_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split PDB complexes manifest by structure.

    Args:
        manifest_path: Path to PDB complexes manifest TSV.
        val_fraction: Fraction of structures assigned to validation.
        seed: Random seed for deterministic splitting.
        structure_column: Column containing PDB IDs.

    Returns:
        Tuple of (train_df, val_df).
    """
    df = pd.read_csv(manifest_path, sep="\t")
    df.columns = df.columns.str.strip()

    col = structure_column
    if col not in df.columns:
        candidates = [c for c in df.columns if "pdb" in c.lower() or "structure" in c.lower()]
        if candidates:
            col = candidates[0]
            logger.info("Using structure column: %s", col)
        else:
            msg = f"No structure column found in {manifest_path}. Columns: {list(df.columns)}"
            raise ValueError(msg)

    unique_ids = df[col].unique()
    val_ids = {c for c in unique_ids if _hash_split(str(c), val_fraction, seed)}

    is_val = df[col].isin(val_ids)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    logger.info(
        "PDB split: %d structures → %d train (%d rows), %d val (%d rows)",
        len(unique_ids),
        len(unique_ids) - len(val_ids),
        len(train_df),
        len(val_ids),
        len(val_df),
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Unified manifest preparation
# ---------------------------------------------------------------------------


def _normalize_to_training_manifest(
    df: pd.DataFrame,
    source_name: str,
    structure_path_column: str = "structure_path",
    chain_a_column: str = "chain_A",
    chain_b_column: str = "chain_B",
) -> pd.DataFrame:
    """Normalize a source-specific DataFrame into the training manifest format.

    The training manifest requires columns: structure_path, chain_A, chain_B, source.
    Source-specific DataFrames may have different column names, but they must
    identify both partner chain groups.

    Args:
        df: Source-specific DataFrame.
        source_name: Name of the data source (teddymer, nvidia, pdb).
        structure_path_column: Column containing structure file paths.
        chain_a_column: Column containing chain A IDs.
        chain_b_column: Column containing chain B IDs.

    Returns:
        DataFrame with standardized training manifest columns.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    # Find structure path column
    path_col = None
    for candidate in [structure_path_column, "structure_path", "path", "filename"]:
        if candidate.lower() in cols:
            path_col = cols[candidate.lower()]
            break
    if path_col is None:
        msg = f"No structure path column found in {source_name} manifest"
        raise ValueError(msg)

    # Find chain columns.
    a_col = cols.get(chain_a_column.lower(), cols.get("chain_a"))
    b_col = cols.get(chain_b_column.lower(), cols.get("chain_b"))
    if a_col is None or b_col is None:
        msg = (
            f"No partner chain columns found in {source_name} manifest; "
            "expected chain_A and chain_B"
        )
        raise ValueError(msg)

    source_id_col = None
    for candidate in ["source_id", "cluster_rep", "cluster_id", "model_id", "pdb_id", "id"]:
        if candidate in cols:
            source_id_col = cols[candidate]
            break

    split_group_col = None
    for candidate in ["split_group", "cluster_rep", "cluster_id", "model_id", "pdb_id"]:
        if candidate in cols:
            split_group_col = cols[candidate]
            break

    interface_col = None
    for candidate in ["interface_residues", "interfacelength", "interface_length"]:
        if candidate in cols:
            interface_col = cols[candidate]
            break

    result = pd.DataFrame()
    result["structure_path"] = df[path_col]
    result["chain_A"] = df[a_col]
    result["chain_B"] = df[b_col]
    result["source"] = source_name
    if source_id_col is not None:
        result["source_id"] = df[source_id_col].astype(str)
    else:
        result["source_id"] = [f"{source_name}_{i}" for i in range(len(df))]
    if split_group_col is not None:
        result["split_group"] = df[split_group_col].astype(str)
    else:
        result["split_group"] = result["source_id"]
    if interface_col is not None:
        result["interface_residues"] = df[interface_col].fillna(0).astype(int)
    else:
        result["interface_residues"] = 0

    return result[list(MANIFEST_COLUMNS)]


def prepare_manifests(
    output_dir: str | Path,
    *,
    teddymer_manifest: str | Path | None = None,
    nvidia_manifest: str | Path | None = None,
    pdb_manifest: str | Path | None = None,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Prepare source-specific and aggregate train/val manifests.

    Splits each source independently, writes source-specific files named
    ``train_<source>.tsv`` and ``val_<source>.tsv``, then concatenates the
    splits into aggregate ``train_manifest.tsv`` and ``val_manifest.tsv`` files.

    Args:
        output_dir: Directory to write output manifests.
        teddymer_manifest: Path to filtered teddymer manifest TSV.
        nvidia_manifest: Path to filtered NVIDIA complexes manifest TSV.
        pdb_manifest: Path to PDB complexes manifest TSV.
        val_fraction: Fraction reserved for validation (default 5%).
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_manifest_path, val_manifest_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    source_paths: list[tuple[str, Path, Path, int, int]] = []

    if teddymer_manifest is not None:
        train_df, val_df = split_teddymer_manifest(
            teddymer_manifest, val_fraction=val_fraction, seed=seed
        )
        train_norm = _normalize_to_training_manifest(train_df, "teddymer")
        val_norm = _normalize_to_training_manifest(val_df, "teddymer")
        train_parts.append(train_norm)
        val_parts.append(val_norm)
        train_source_path = output_dir / "train_teddymer.tsv"
        val_source_path = output_dir / "val_teddymer.tsv"
        train_norm.to_csv(train_source_path, sep="\t", index=False)
        val_norm.to_csv(val_source_path, sep="\t", index=False)
        source_paths.append(
            ("teddymer", train_source_path, val_source_path, len(train_norm), len(val_norm))
        )

    if nvidia_manifest is not None:
        train_df, val_df = split_nvidia_manifest(
            nvidia_manifest, val_fraction=val_fraction, seed=seed
        )
        train_norm = _normalize_to_training_manifest(train_df, "nvidia")
        val_norm = _normalize_to_training_manifest(val_df, "nvidia")
        train_parts.append(train_norm)
        val_parts.append(val_norm)
        train_source_path = output_dir / "train_nvidia.tsv"
        val_source_path = output_dir / "val_nvidia.tsv"
        train_norm.to_csv(train_source_path, sep="\t", index=False)
        val_norm.to_csv(val_source_path, sep="\t", index=False)
        source_paths.append(
            ("nvidia", train_source_path, val_source_path, len(train_norm), len(val_norm))
        )

    if pdb_manifest is not None:
        train_df, val_df = split_pdb_manifest(pdb_manifest, val_fraction=val_fraction, seed=seed)
        train_norm = _normalize_to_training_manifest(train_df, "pdb")
        val_norm = _normalize_to_training_manifest(val_df, "pdb")
        train_parts.append(train_norm)
        val_parts.append(val_norm)
        train_source_path = output_dir / "train_pdb.tsv"
        val_source_path = output_dir / "val_pdb.tsv"
        train_norm.to_csv(train_source_path, sep="\t", index=False)
        val_norm.to_csv(val_source_path, sep="\t", index=False)
        source_paths.append(
            ("pdb", train_source_path, val_source_path, len(train_norm), len(val_norm))
        )

    if not train_parts:
        msg = "At least one data source manifest must be provided"
        raise ValueError(msg)

    train_manifest = pd.concat(train_parts, ignore_index=True)
    val_manifest = pd.concat(val_parts, ignore_index=True)

    train_path = output_dir / "train_manifest.tsv"
    val_path = output_dir / "val_manifest.tsv"

    train_manifest.to_csv(train_path, sep="\t", index=False)
    val_manifest.to_csv(val_path, sep="\t", index=False)

    logger.info(
        "Wrote unified manifests: train=%d rows (%s), val=%d rows (%s)",
        len(train_manifest),
        train_path,
        len(val_manifest),
        val_path,
    )

    # Write split statistics
    stats_path = output_dir / "split_stats.tsv"
    stats_rows = []
    for source, train_source_path, val_source_path, n_train, n_val in source_paths:
        stats_rows.append(
            {
                "source": source,
                "n_train": n_train,
                "n_val": n_val,
                "total": n_train + n_val,
                "val_pct": f"{100 * n_val / max(n_train + n_val, 1):.1f}%",
                "train_manifest": str(train_source_path),
                "val_manifest": str(val_source_path),
            }
        )
    pd.DataFrame(stats_rows).to_csv(stats_path, sep="\t", index=False)
    logger.info("Wrote split statistics to %s", stats_path)

    return train_path, val_path
