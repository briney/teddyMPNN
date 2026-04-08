"""Interface sequence recovery metrics.

Evaluates how well a model recovers native amino acid sequences at
protein-protein interface positions using teacher-forced predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from teddympnn.data.features import identify_interface_residues

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from teddympnn.models.protein_mpnn import ProteinMPNN

logger = logging.getLogger(__name__)

# Interface size bin boundaries (number of interface residues)
_SIZE_BINS: list[tuple[str, int, int]] = [
    ("small", 1, 20),
    ("medium", 21, 50),
    ("large", 51, 100_000),
]


@dataclass
class RecoveryResults:
    """Sequence recovery evaluation results.

    Attributes:
        overall_recovery: Micro-averaged recovery across all designed positions.
        interface_recovery: Micro-averaged recovery at interface positions only.
        per_structure_recovery: Macro-averaged (mean of per-structure recoveries).
        per_structure_interface_recovery: Macro-averaged interface recovery.
        n_structures: Number of structures evaluated.
        n_designed_residues: Total designed residues scored.
        n_interface_residues: Total interface residues scored.
        size_bin_recoveries: Interface recovery stratified by interface size bin.
    """

    overall_recovery: float
    interface_recovery: float
    per_structure_recovery: float
    per_structure_interface_recovery: float
    n_structures: int
    n_designed_residues: int
    n_interface_residues: int
    size_bin_recoveries: dict[str, float] = field(default_factory=dict)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move batch tensors to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def compute_recovery(
    model: ProteinMPNN,
    data_loader: DataLoader[dict[str, Any]],
    interface_cutoff: float = 8.0,
    device: torch.device | None = None,
) -> RecoveryResults:
    """Compute sequence recovery metrics on a dataset.

    For each structure, runs teacher-forcing to get argmax predictions and
    compares to ground truth at designed positions. Also computes
    interface-only recovery using CB-CB distance to the partner chain.

    Args:
        model: ProteinMPNN or LigandMPNN model.
        data_loader: DataLoader yielding batched feature dicts.
        interface_cutoff: CB-CB distance cutoff for interface residues (A).
        device: Device for computation. Defaults to model device.

    Returns:
        RecoveryResults with overall, interface, and per-structure metrics.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # Per-structure accumulators
    structure_recoveries: list[float] = []
    structure_iface_recoveries: list[float] = []
    structure_iface_sizes: list[int] = []

    # Global accumulators
    total_correct = 0
    total_designed = 0
    total_iface_correct = 0
    total_iface = 0

    # Per-size-bin accumulators
    bin_correct: dict[str, int] = {name: 0 for name, _, _ in _SIZE_BINS}
    bin_total: dict[str, int] = {name: 0 for name, _, _ in _SIZE_BINS}

    for batch in data_loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        preds = output["log_probs"].argmax(dim=-1)  # (B, L)

        B = preds.shape[0]
        for b in range(B):
            res_mask = batch["residue_mask"][b].bool()
            designed_mask = batch["designed_residue_mask"][b].bool()
            valid_designed = designed_mask & res_mask

            n_valid = res_mask.sum().item()
            if n_valid == 0:
                continue

            # Compute interface mask on valid residues only
            xyz_37_valid = batch["xyz_37"][b][res_mask]
            xyz_37_m_valid = batch["xyz_37_m"][b][res_mask]
            chain_labels_valid = batch["chain_labels"][b][res_mask]

            iface_on_valid = identify_interface_residues(
                xyz_37_valid,
                xyz_37_m_valid,
                chain_labels_valid,
                distance_cutoff=interface_cutoff,
            )
            # Map back to padded space
            full_iface = torch.zeros_like(res_mask)
            full_iface[res_mask] = iface_on_valid
            designed_iface = valid_designed & full_iface

            correct = preds[b] == batch["S"][b]

            # Overall designed recovery
            n_des = valid_designed.sum().item()
            n_cor = (correct & valid_designed).sum().item()
            total_correct += n_cor
            total_designed += n_des
            if n_des > 0:
                structure_recoveries.append(n_cor / n_des)

            # Interface recovery
            n_if = designed_iface.sum().item()
            n_if_cor = (correct & designed_iface).sum().item()
            total_iface_correct += n_if_cor
            total_iface += n_if
            structure_iface_sizes.append(n_if)

            if n_if > 0:
                structure_iface_recoveries.append(n_if_cor / n_if)

                # Size bin stratification
                for name, lo, hi in _SIZE_BINS:
                    if lo <= n_if <= hi:
                        bin_correct[name] += n_if_cor
                        bin_total[name] += n_if
                        break

    # Aggregate
    size_bin_recoveries = {
        name: bin_correct[name] / max(bin_total[name], 1)
        for name, _, _ in _SIZE_BINS
        if bin_total[name] > 0
    }

    return RecoveryResults(
        overall_recovery=total_correct / max(total_designed, 1),
        interface_recovery=total_iface_correct / max(total_iface, 1),
        per_structure_recovery=(sum(structure_recoveries) / max(len(structure_recoveries), 1)),
        per_structure_interface_recovery=(
            sum(structure_iface_recoveries) / max(len(structure_iface_recoveries), 1)
        ),
        n_structures=len(structure_recoveries),
        n_designed_residues=total_designed,
        n_interface_residues=total_iface,
        size_bin_recoveries=size_bin_recoveries,
    )
