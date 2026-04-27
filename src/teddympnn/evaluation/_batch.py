"""Shared structure-to-batch construction utilities for evaluation.

These helpers are reused by ``score_structure``, ``score_complex``,
``predict_ddg``, and ``evaluate_skempi`` so that ProteinMPNN and LigandMPNN
inputs are built consistently. In particular, LigandMPNN batches always
include ligand atom tensors (``Y``/``Y_m``/``Y_t``); ProteinMPNN batches
omit them since the model does not consume them.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime in load_eval_features signature
from typing import Any

import torch

from teddympnn.data.features import (
    derive_backbone,
    extract_ligand_atoms,
    extract_sidechain_atoms,
    parse_structure,
)


def load_eval_features(
    structure_path: str | Path,
    *,
    model_type: str,
) -> dict[str, Any]:
    """Parse a structure file and assemble model-aware features.

    Always returns ``chain_ids``, ``residue_numbers``, ``residue_icodes``.
    For ``model_type == "ligand_mpnn"``, also merges in non-protein
    ligand atoms via ``extract_ligand_atoms``.

    Args:
        structure_path: Path to a PDB or mmCIF file.
        model_type: ``"protein_mpnn"`` or ``"ligand_mpnn"``.

    Returns:
        Unbatched feature dict.
    """
    features = parse_structure(structure_path)
    if model_type == "ligand_mpnn":
        features.update(extract_ligand_atoms(structure_path))
    return features


def extract_chain_view(
    features: dict[str, Any],
    chain_ids: list[str],
    target_chains: set[str],
    *,
    residue_numbers: list[int] | None = None,
    residue_icodes: list[str] | None = None,
) -> tuple[dict[str, Any], list[str], list[int], list[str]]:
    """Filter unbatched features down to a chain subset.

    Ligand fields (``Y``/``Y_m``/``Y_t``) are passed through unchanged —
    LigandMPNN context is global to the structure, not per chain.

    Args:
        features: Unbatched feature dict from ``load_eval_features``.
        chain_ids: Per-residue chain ID strings.
        target_chains: Set of chain IDs to retain.
        residue_numbers: Per-residue PDB residue numbers (defaults to
            ``features["residue_numbers"]``).
        residue_icodes: Per-residue insertion codes (defaults to
            ``features["residue_icodes"]``).

    Returns:
        ``(filtered_features, filtered_chain_ids, filtered_residue_numbers,
        filtered_residue_icodes)``.
    """
    if residue_numbers is None:
        residue_numbers = features.get("residue_numbers", [])
    if residue_icodes is None:
        residue_icodes = features.get("residue_icodes", [""] * len(chain_ids))

    mask = torch.tensor([cid in target_chains for cid in chain_ids])
    new_features: dict[str, Any] = {}
    for key in ("xyz_37", "xyz_37_m", "S", "R_idx", "chain_labels", "residue_mask"):
        if key in features:
            new_features[key] = features[key][mask]

    for key in ("Y", "Y_m", "Y_t"):
        if key in features:
            new_features[key] = features[key]

    mask_list = mask.tolist()
    new_chain_ids = [cid for cid, m in zip(chain_ids, mask_list, strict=True) if m]
    new_residue_numbers = [rn for rn, m in zip(residue_numbers, mask_list, strict=True) if m]
    new_residue_icodes = [ic for ic, m in zip(residue_icodes, mask_list, strict=True) if m]
    return new_features, new_chain_ids, new_residue_numbers, new_residue_icodes


def build_eval_batch(
    features: dict[str, Any],
    designed_mask: torch.Tensor,
    device: torch.device,
    *,
    model_type: str,
    fixed_residue_mask: torch.Tensor | None = None,
    include_partner_sidechains: bool = True,
) -> dict[str, torch.Tensor]:
    """Create a single-example (B=1) batch from unbatched features.

    For LigandMPNN, the returned batch always includes ``Y``/``Y_m``/``Y_t``,
    composed of the parsed non-protein atoms plus (optionally) the
    fixed-partner side-chain atoms.

    Args:
        features: Unbatched feature dict (see ``load_eval_features``).
        designed_mask: ``(L,)`` bool — True at designed-partner residues.
        device: Device to place batch tensors on.
        model_type: ``"protein_mpnn"`` or ``"ligand_mpnn"``.
        fixed_residue_mask: ``(L,)`` bool — True at fixed-partner residues
            whose side chains should be atomized. Defaults to
            ``~designed_mask`` when ``include_partner_sidechains`` is True.
        include_partner_sidechains: Whether to atomize the fixed partner's
            side chains into the LigandMPNN context (no effect for
            ProteinMPNN).

    Returns:
        Batched feature dict (B=1) ready for ``model(...)``/``model.score``.
    """
    X, _ = derive_backbone(features["xyz_37"], features["xyz_37_m"])
    fixed_mask = fixed_residue_mask if fixed_residue_mask is not None else ~designed_mask
    batch: dict[str, torch.Tensor] = {
        "X": X.unsqueeze(0).to(device),
        "S": features["S"].unsqueeze(0).to(device),
        "R_idx": features["R_idx"].unsqueeze(0).to(device),
        "chain_labels": features["chain_labels"].unsqueeze(0).to(device),
        "residue_mask": features["residue_mask"].unsqueeze(0).to(device),
        "designed_residue_mask": designed_mask.unsqueeze(0).to(device),
        "fixed_residue_mask": fixed_mask.unsqueeze(0).to(device),
    }

    if model_type == "ligand_mpnn":
        Y = features.get("Y", torch.zeros(0, 3, dtype=torch.float32))
        Y_m = features.get("Y_m", torch.zeros(0, dtype=torch.bool))
        Y_t = features.get("Y_t", torch.zeros(0, dtype=torch.long))
        if include_partner_sidechains and fixed_mask.any():
            sc = extract_sidechain_atoms(
                features["xyz_37"],
                features["xyz_37_m"],
                features["S"],
                fixed_mask,
            )
            if sc["Y"].shape[0] > 0:
                Y = torch.cat([Y, sc["Y"]], dim=0)
                Y_m = torch.cat([Y_m, sc["Y_m"]], dim=0)
                Y_t = torch.cat([Y_t, sc["Y_t"]], dim=0)
        batch["Y"] = Y.unsqueeze(0).to(device)
        batch["Y_m"] = Y_m.unsqueeze(0).to(device)
        batch["Y_t"] = Y_t.unsqueeze(0).to(device)

    return batch
