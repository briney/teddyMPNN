"""SKEMPI v2.0 benchmark for binding affinity prediction.

Downloads and parses the SKEMPI v2.0 database, runs ddG predictions
for each entry, and computes correlation metrics against experimental
values.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from teddympnn.evaluation.binding_affinity import predict_ddg

if TYPE_CHECKING:
    from teddympnn.models.protein_mpnn import ProteinMPNN

logger = logging.getLogger(__name__)

SKEMPI_CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Regex for SKEMPI mutation strings: chain + wt_aa + resnum(+icode) + mut_aa
_MUTATION_RE = re.compile(r"^([A-Za-z])([A-Z])(-?\d+[A-Za-z]?)([A-Z])$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SKEMPIEntry:
    """A single SKEMPI v2.0 measurement.

    Attributes:
        pdb_id: 4-letter PDB code.
        partner_chains: Tuple of (partner_1_chain_ids, partner_2_chain_ids).
        mutations: ``{chain_id: {mutation_str: None}}`` for predict_ddg.
        ddg_experimental: Experimental ddG in kcal/mol.
        mutation_type: ``"single"`` or ``"multi"``.
    """

    pdb_id: str
    partner_chains: tuple[set[str], set[str]]
    mutations: dict[str, dict[str, str | None]]
    ddg_experimental: float
    mutation_type: str


@dataclass
class SKEMPIResults:
    """Aggregate SKEMPI v2.0 benchmark results.

    Attributes:
        spearman: Overall Spearman rank correlation.
        pearson: Overall Pearson correlation.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        auroc: Area under ROC for classifying stabilizing (ddG < 0) vs
            destabilizing mutations.
        per_structure_spearman_median: Median per-structure Spearman correlation.
        n_entries: Total entries evaluated.
        n_structures: Number of unique PDB structures.
        per_structure_spearman: Per-structure Spearman values.
    """

    spearman: float
    pearson: float
    rmse: float
    mae: float
    auroc: float
    per_structure_spearman_median: float
    n_entries: int
    n_structures: int
    per_structure_spearman: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statistics (numpy-only, no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Rank data with averaged ties."""
    n = len(x)
    sorter = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = np.arange(1, n + 1, dtype=np.float64)

    # Average tied ranks
    sorted_x = x[sorter]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[sorter[i:j]])
            ranks[sorter[i:j]] = avg_rank
        i = j
    return ranks


def pearson_correlation(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """Pearson correlation coefficient."""
    if len(x) < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    if denom == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman_correlation(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    """Spearman rank correlation."""
    if len(x) < 2:
        return float("nan")
    return pearson_correlation(_rankdata(x), _rankdata(y))


def auroc(y_true: np.ndarray[Any, Any], y_scores: np.ndarray[Any, Any]) -> float:
    """Area under the ROC curve via Mann-Whitney U statistic.

    Args:
        y_true: Binary labels (1 = positive).
        y_scores: Predicted scores (higher = more likely positive).
    """
    pos = y_scores[y_true.astype(bool)]
    neg = y_scores[~y_true.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Count concordant pairs via broadcasting
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


# ---------------------------------------------------------------------------
# SKEMPI parsing
# ---------------------------------------------------------------------------


def _parse_mutation_string(mut_str: str) -> tuple[str, str]:
    """Parse a single SKEMPI mutation field like ``'IA45G'``.

    Returns:
        Tuple of (chain_id, mutation_str) where mutation_str is ``"A45G"``.
    """
    m = _MUTATION_RE.match(mut_str.strip())
    if m is None:
        msg = f"Cannot parse SKEMPI mutation: '{mut_str}'"
        raise ValueError(msg)
    chain_id = m.group(1)
    wt_aa = m.group(2)
    resnum = m.group(3)
    mut_aa = m.group(4)
    return chain_id, f"{wt_aa}{resnum}{mut_aa}"


def _parse_partner_chains(pdb_field: str) -> tuple[str, tuple[set[str], set[str]]]:
    """Parse the SKEMPI ``#Pdb`` field like ``'1BRS_A_D'``.

    Returns:
        Tuple of (pdb_id, (partner_1_chains, partner_2_chains)).
    """
    parts = pdb_field.strip().split("_")
    pdb_id = parts[0].upper()
    # Remaining parts are chain groups — SKEMPI uses _ to separate partners
    # Partner 1 is the second field, Partner 2 is the third
    if len(parts) >= 3:
        p1 = set(parts[1])
        p2 = set(parts[2])
    elif len(parts) == 2:
        p1 = set(parts[1])
        p2 = set()
    else:
        p1 = set()
        p2 = set()
    return pdb_id, (p1, p2)


def parse_skempi(csv_path: str | Path) -> list[SKEMPIEntry]:
    """Parse a SKEMPI v2.0 CSV file into structured entries.

    Args:
        csv_path: Path to the SKEMPI v2.0 CSV file (semicolon-delimited).

    Returns:
        List of SKEMPIEntry objects with valid ddG values and parseable mutations.
    """
    csv_path = Path(csv_path)
    entries: list[SKEMPIEntry] = []
    skipped = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            # Skip entries without experimental ddG
            ddg_str = row.get("ddG", "").strip()
            if not ddg_str or ddg_str == "":
                skipped += 1
                continue
            try:
                ddg_val = float(ddg_str)
            except ValueError:
                skipped += 1
                continue

            # Parse PDB and partner chains
            pdb_field = row.get("#Pdb", "").strip()
            if not pdb_field:
                skipped += 1
                continue

            try:
                pdb_id, partner_chains = _parse_partner_chains(pdb_field)
            except Exception:
                skipped += 1
                continue

            # Parse mutations
            mut_field = row.get("Mutation(s)_PDB", "").strip()
            if not mut_field:
                skipped += 1
                continue

            try:
                mutations: dict[str, dict[str, str | None]] = {}
                mut_parts = mut_field.split(",")
                for part in mut_parts:
                    chain_id, mut_str = _parse_mutation_string(part)
                    if chain_id not in mutations:
                        mutations[chain_id] = {}
                    mutations[chain_id][mut_str] = None
            except ValueError:
                skipped += 1
                continue

            mut_type = "single" if len(mut_parts) == 1 else "multi"

            entries.append(
                SKEMPIEntry(
                    pdb_id=pdb_id,
                    partner_chains=partner_chains,
                    mutations=mutations,
                    ddg_experimental=ddg_val,
                    mutation_type=mut_type,
                )
            )

    logger.info("Parsed %d SKEMPI entries (%d skipped)", len(entries), skipped)
    return entries


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def download_skempi(output_dir: str | Path) -> Path:
    """Download the SKEMPI v2.0 CSV file.

    Args:
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "skempi_v2.csv"

    if csv_path.exists():
        logger.info("SKEMPI CSV already exists at %s", csv_path)
        return csv_path

    logger.info("Downloading SKEMPI v2.0 from %s", SKEMPI_CSV_URL)
    torch.hub.download_url_to_file(SKEMPI_CSV_URL, str(csv_path))
    logger.info("Downloaded SKEMPI CSV to %s", csv_path)
    return csv_path


def download_pdb_structure(pdb_id: str, output_dir: str | Path) -> Path:
    """Download a PDB structure from RCSB.

    Args:
        pdb_id: 4-letter PDB code.
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded PDB file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"{pdb_id.upper()}.pdb"

    if pdb_path.exists():
        return pdb_path

    url = f"{RCSB_DOWNLOAD_URL}/{pdb_id.upper()}.pdb"
    logger.info("Downloading %s", url)
    torch.hub.download_url_to_file(url, str(pdb_path))
    return pdb_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_skempi(
    model: ProteinMPNN,
    skempi_dir: str | Path,
    num_samples: int = 20,
    structure_noise: float = 0.0,
    device: torch.device | None = None,
    max_entries: int | None = None,
) -> SKEMPIResults:
    """Run the full SKEMPI v2.0 benchmark.

    Downloads (if needed) the SKEMPI CSV and PDB structures, runs ddG
    predictions for each entry, and computes correlation metrics against
    experimental values.

    Args:
        model: ProteinMPNN or LigandMPNN model.
        skempi_dir: Directory containing (or to download) SKEMPI data.
        num_samples: Monte Carlo samples per ddG prediction.
        structure_noise: Backbone noise for ensemble scoring (A).
        device: Device for computation.
        max_entries: Limit evaluation to first N entries (for testing).

    Returns:
        SKEMPIResults with correlation metrics.
    """
    skempi_dir = Path(skempi_dir)
    structures_dir = skempi_dir / "structures"

    # Download SKEMPI CSV
    csv_path = download_skempi(skempi_dir)
    entries = parse_skempi(csv_path)

    if max_entries is not None:
        entries = entries[:max_entries]

    # Predict ddG for each entry
    predictions: list[float] = []
    experimentals: list[float] = []
    pdb_ids: list[str] = []
    skipped = 0

    for entry in entries:
        pdb_path = structures_dir / f"{entry.pdb_id}.pdb"
        if not pdb_path.exists():
            try:
                pdb_path = download_pdb_structure(entry.pdb_id, structures_dir)
            except Exception:
                logger.warning("Failed to download %s, skipping", entry.pdb_id)
                skipped += 1
                continue

        try:
            pred = predict_ddg(
                model,
                pdb_path,
                entry.mutations,
                num_samples=num_samples,
                structure_noise=structure_noise,
                partner_chains=entry.partner_chains,
                device=device,
            )
        except Exception:
            logger.warning(
                "Failed to predict ddG for %s %s, skipping",
                entry.pdb_id,
                entry.mutations,
                exc_info=True,
            )
            skipped += 1
            continue

        predictions.append(pred)
        experimentals.append(entry.ddg_experimental)
        pdb_ids.append(entry.pdb_id)

    if skipped > 0:
        logger.info("Skipped %d entries due to errors", skipped)

    if len(predictions) < 2:
        logger.warning("Too few successful predictions (%d) for metrics", len(predictions))
        return SKEMPIResults(
            spearman=float("nan"),
            pearson=float("nan"),
            rmse=float("nan"),
            mae=float("nan"),
            auroc=float("nan"),
            per_structure_spearman_median=float("nan"),
            n_entries=len(predictions),
            n_structures=len(set(pdb_ids)),
        )

    pred_arr = np.array(predictions)
    exp_arr = np.array(experimentals)
    pdb_arr = np.array(pdb_ids)

    # Overall metrics
    overall_spearman = spearman_correlation(exp_arr, pred_arr)
    overall_pearson = pearson_correlation(exp_arr, pred_arr)
    overall_rmse = float(np.sqrt(np.mean((exp_arr - pred_arr) ** 2)))
    overall_mae = float(np.mean(np.abs(exp_arr - pred_arr)))

    # AUROC: stabilizing (ddG < 0) vs destabilizing
    labels = (exp_arr < 0).astype(np.float64)
    # Predicted ddG < 0 should indicate stabilizing → use negative pred as score
    overall_auroc = auroc(labels, -pred_arr)

    # Per-structure Spearman
    per_struct_spearman: dict[str, float] = {}
    for pdb_id in set(pdb_ids):
        mask = pdb_arr == pdb_id
        if mask.sum() >= 3:
            sp = spearman_correlation(exp_arr[mask], pred_arr[mask])
            per_struct_spearman[pdb_id] = sp

    median_spearman = float("nan")
    if per_struct_spearman:
        vals = [v for v in per_struct_spearman.values() if not np.isnan(v)]
        if vals:
            median_spearman = float(np.median(vals))

    return SKEMPIResults(
        spearman=overall_spearman,
        pearson=overall_pearson,
        rmse=overall_rmse,
        mae=overall_mae,
        auroc=overall_auroc,
        per_structure_spearman_median=median_spearman,
        n_entries=len(predictions),
        n_structures=len(set(pdb_ids)),
        per_structure_spearman=per_struct_spearman,
    )
