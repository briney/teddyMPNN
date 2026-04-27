"""Binding affinity (ddG) prediction via thermodynamic decomposition.

Implements the StaB-ddG / BA-DDG approach: score wild-type and mutant
sequences on the complex and each monomer, then compute ddG from the
difference in binding energy proxies.

Each ddG prediction requires 6 forward passes per Monte Carlo sample
(wt/mut x complex/chain_A/chain_B), with antithetic variates (shared
random decoding order) for variance reduction.
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime in predict_ddg signature
from typing import Any

import torch

from teddympnn.data.features import (
    derive_backbone,
    extract_ligand_atoms,
    extract_sidechain_atoms,
    parse_structure,
)
from teddympnn.models.ligand_mpnn import LigandMPNN
from teddympnn.models.tokens import (
    AMINO_ACIDS_1TO3,
    idx_to_token,
    token_to_idx,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_batch(
    features: dict[str, Any],
    designed_mask: torch.Tensor,
    device: torch.device,
    *,
    include_ligand_context: bool = False,
) -> dict[str, torch.Tensor]:
    """Create a single-example batch (B=1) from unbatched features."""
    X, X_m = derive_backbone(features["xyz_37"], features["xyz_37_m"])
    batch = {
        "X": X.unsqueeze(0).to(device),
        "S": features["S"].unsqueeze(0).to(device),
        "R_idx": features["R_idx"].unsqueeze(0).to(device),
        "chain_labels": features["chain_labels"].unsqueeze(0).to(device),
        "residue_mask": features["residue_mask"].unsqueeze(0).to(device),
        "designed_residue_mask": designed_mask.unsqueeze(0).to(device),
        "fixed_residue_mask": (~designed_mask).unsqueeze(0).to(device),
    }
    if include_ligand_context:
        Y = features.get("Y", torch.zeros(0, 3, dtype=torch.float32))
        Y_m = features.get("Y_m", torch.zeros(0, dtype=torch.bool))
        Y_t = features.get("Y_t", torch.zeros(0, dtype=torch.long))
        sidechains = extract_sidechain_atoms(
            features["xyz_37"],
            features["xyz_37_m"],
            features["S"],
            ~designed_mask,
        )
        if sidechains["Y"].shape[0] > 0:
            Y = torch.cat([Y, sidechains["Y"]], dim=0)
            Y_m = torch.cat([Y_m, sidechains["Y_m"]], dim=0)
            Y_t = torch.cat([Y_t, sidechains["Y_t"]], dim=0)
        batch["Y"] = Y.unsqueeze(0).to(device)
        batch["Y_m"] = Y_m.unsqueeze(0).to(device)
        batch["Y_t"] = Y_t.unsqueeze(0).to(device)
    return batch


def _extract_chain_features(
    features: dict[str, Any],
    chain_ids: list[str],
    target_chains: set[str],
) -> tuple[dict[str, Any], list[str], list[int]]:
    """Extract features for a subset of chains.

    Args:
        features: Unbatched feature dict from parse_structure.
        chain_ids: Per-residue chain ID strings.
        target_chains: Set of chain IDs to keep.

    Returns:
        Tuple of (filtered features, filtered chain_ids, filtered residue_numbers).
    """
    mask = torch.tensor([cid in target_chains for cid in chain_ids])
    new_features: dict[str, Any] = {}
    for key in ("xyz_37", "xyz_37_m", "S", "R_idx", "chain_labels", "residue_mask"):
        if key in features:
            new_features[key] = features[key][mask]

    mask_list = mask.tolist()
    new_chain_ids = [cid for cid, m in zip(chain_ids, mask_list, strict=True) if m]
    residue_numbers: list[int] = features.get("residue_numbers", [])
    new_residue_numbers = [rn for rn, m in zip(residue_numbers, mask_list, strict=True) if m]

    return new_features, new_chain_ids, new_residue_numbers


def _apply_mutations(
    features: dict[str, Any],
    chain_ids: list[str],
    residue_numbers: list[int],
    mutations: dict[str, dict[str, str | None]],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply mutations to the sequence tensor.

    Args:
        features: Unbatched feature dict.
        chain_ids: Per-residue chain IDs.
        residue_numbers: Per-residue PDB residue numbers.
        mutations: ``{chain_id: {mutation_str: None}}`` where mutation_str
            is ``"A45G"`` (wt_1letter + resnum + mut_1letter).

    Returns:
        Tuple of (mutant features, mutation_mask as ``(L,)`` bool tensor).
    """
    S_mut = features["S"].clone()
    mutation_mask = torch.zeros(len(S_mut), dtype=torch.bool)

    for chain_id, chain_muts in mutations.items():
        for mut_str in chain_muts:
            wt_aa_1 = mut_str[0]
            mut_aa_1 = mut_str[-1]
            resnum = int(mut_str[1:-1])

            wt_aa_3 = AMINO_ACIDS_1TO3.get(wt_aa_1)
            mut_aa_3 = AMINO_ACIDS_1TO3.get(mut_aa_1)
            if wt_aa_3 is None or mut_aa_3 is None:
                msg = f"Unknown amino acid in mutation '{mut_str}'"
                raise ValueError(msg)

            found = False
            for i, (cid, rn) in enumerate(zip(chain_ids, residue_numbers, strict=True)):
                if cid == chain_id and rn == resnum:
                    expected_idx = token_to_idx[wt_aa_3]
                    actual_idx = features["S"][i].item()
                    if actual_idx != expected_idx:
                        actual_name = idx_to_token.get(actual_idx, "???")
                        logger.warning(
                            "Mutation %s on chain %s: expected %s at position %d but found %s",
                            mut_str,
                            chain_id,
                            wt_aa_3,
                            resnum,
                            actual_name,
                        )
                    S_mut[i] = token_to_idx[mut_aa_3]
                    mutation_mask[i] = True
                    found = True
                    break

            if not found:
                msg = f"Residue {chain_id}:{resnum} not found in structure"
                raise ValueError(msg)

    mut_features = {**features, "S": S_mut}
    return mut_features, mutation_mask


def _map_mask_to_chain(
    mutation_mask: torch.Tensor,
    chain_ids: list[str],
    target_chains: set[str],
) -> torch.Tensor:
    """Map a complex-level mutation mask to a chain-level mask.

    Returns a mask of length equal to the number of residues in target_chains,
    with True at positions that correspond to mutation sites.
    """
    chain_member = torch.tensor([cid in target_chains for cid in chain_ids])
    # Indices of target chain residues in the complex
    complex_indices = chain_member.nonzero(as_tuple=True)[0]
    return mutation_mask[complex_indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_structure(
    model: torch.nn.Module,
    input_features: dict[str, torch.Tensor],
    score_mask: torch.Tensor,
    seed: int,
    structure_noise: float = 0.0,
) -> float:
    """Score a structure with a deterministic random state.

    Sets the random seed before scoring so that the decoding order (and
    optionally backbone noise) are reproducible. For antithetic variates
    in ddG prediction, call with the same seed for wild-type and mutant.

    Args:
        model: ProteinMPNN or LigandMPNN model (set to eval externally).
        input_features: Batched feature dict (B=1).
        score_mask: ``(1, L)`` mask of positions to score.
        seed: Random seed for reproducible decoding order.
        structure_noise: Gaussian noise std for backbone coordinates (A).

    Returns:
        Sum of log-probabilities at masked positions.
    """
    device = input_features["X"].device

    # Set seed for reproducible noise + decoding order
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    features = input_features
    if structure_noise > 0:
        noise = torch.randn_like(input_features["X"])
        features = {**input_features, "X": input_features["X"] + structure_noise * noise}

    # model.score() calls _setup_causality_masks which uses torch.rand
    # Since we seeded above and noise consumed a fixed amount of randomness
    # (determined by X.shape, identical for wt and mut), the decoding order
    # will be identical when called with the same seed.
    per_residue: torch.Tensor = model.score(features, score_mask=score_mask)  # type: ignore[operator]
    return float(per_residue.sum().item())


@torch.no_grad()
def predict_ddg(
    model: torch.nn.Module,
    structure_path: str | Path,
    mutations: dict[str, dict[str, str | None]],
    num_samples: int = 20,
    structure_noise: float = 0.0,
    partner_chains: tuple[set[str], set[str]] | None = None,
    device: torch.device | None = None,
) -> float:
    """Predict binding ddG for a set of mutations.

    Uses the thermodynamic decomposition from StaB-ddG / BA-DDG::

        ddG(wt -> mut) = [b_mut - b_wt]
        b = log p(s | X_AB) - log p(s | X_A) - log p(s | X_B)

    where log p is the sum of per-residue log-probabilities restricted to
    the mutation site(s).

    Each Monte Carlo sample uses a fresh random decoding order, with
    antithetic variates (shared randomness for wt vs mut within each
    sample) for variance reduction.

    Args:
        model: ProteinMPNN or LigandMPNN model.
        structure_path: Path to PDB/mmCIF structure file.
        mutations: ``{chain_id: {mutation_str: None}}`` where each
            mutation_str is formatted as ``"A45G"`` (wt + resnum + mut).
        num_samples: Number of Monte Carlo samples for averaging.
        structure_noise: Backbone noise for ensemble scoring (A).
        partner_chains: Explicit partner grouping as
            ``(chain_A_ids, chain_B_ids)``. If None, auto-detected from
            the two unique chain IDs in the structure.
        device: Device for computation.

    Returns:
        Predicted ddG value (kcal/mol proxy).

    Raises:
        ValueError: If partner chains cannot be determined or mutations
            reference unknown residues.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # Parse structure
    features = parse_structure(structure_path)
    is_ligand_model = isinstance(model, LigandMPNN)
    if is_ligand_model:
        features.update(extract_ligand_atoms(structure_path))
    chain_ids: list[str] = features["chain_ids"]
    residue_numbers: list[int] = features["residue_numbers"]

    # Determine partner chain groupings
    if partner_chains is None:
        unique_chains = sorted(set(chain_ids))
        if len(unique_chains) != 2:
            msg = (
                f"Cannot auto-detect partners: found {len(unique_chains)} chains "
                f"({unique_chains}). Provide partner_chains explicitly."
            )
            raise ValueError(msg)
        partner_a = {unique_chains[0]}
        partner_b = {unique_chains[1]}
    else:
        partner_a, partner_b = partner_chains

    # Apply mutations
    mut_features, mutation_mask = _apply_mutations(
        features,
        chain_ids,
        residue_numbers,
        mutations,
    )

    # Extract chain-level views and masks
    chain_a_wt, chain_a_ids, _ = _extract_chain_features(
        features,
        chain_ids,
        partner_a,
    )
    chain_a_mut, _, _ = _extract_chain_features(
        mut_features,
        chain_ids,
        partner_a,
    )
    chain_b_wt, chain_b_ids, _ = _extract_chain_features(
        features,
        chain_ids,
        partner_b,
    )
    chain_b_mut, _, _ = _extract_chain_features(
        mut_features,
        chain_ids,
        partner_b,
    )

    # Build mutation masks for each view
    mask_ab = mutation_mask  # (L_AB,)
    mask_a = _map_mask_to_chain(mutation_mask, chain_ids, partner_a)  # (L_A,)
    mask_b = _map_mask_to_chain(mutation_mask, chain_ids, partner_b)  # (L_B,)

    # Build batches (B=1) — all residues are "designed" for scoring
    L_ab = len(chain_ids)
    L_a = len(chain_a_ids)
    L_b = len(chain_b_ids)

    batch_ab_wt = _make_batch(
        features,
        mask_ab,
        device,
        include_ligand_context=is_ligand_model,
    )
    batch_ab_mut = _make_batch(
        mut_features,
        mask_ab,
        device,
        include_ligand_context=is_ligand_model,
    )
    batch_a_wt = _make_batch(
        chain_a_wt,
        mask_a,
        device,
        include_ligand_context=is_ligand_model,
    )
    batch_a_mut = _make_batch(
        chain_a_mut,
        mask_a,
        device,
        include_ligand_context=is_ligand_model,
    )
    batch_b_wt = _make_batch(
        chain_b_wt,
        mask_b,
        device,
        include_ligand_context=is_ligand_model,
    )
    batch_b_mut = _make_batch(
        chain_b_mut,
        mask_b,
        device,
        include_ligand_context=is_ligand_model,
    )

    # Score masks (on device, shape (1, L))
    smask_ab = mask_ab.unsqueeze(0).to(device)
    smask_a = mask_a.unsqueeze(0).to(device)
    smask_b = mask_b.unsqueeze(0).to(device)

    # Monte Carlo sampling with antithetic variates
    ddg_samples: list[float] = []

    for sample_idx in range(num_samples):
        # Use different seed offsets for each structure view so that
        # the decoding orders are independent across views but paired
        # between wt and mut within each view.
        seed_ab = sample_idx * 3
        seed_a = sample_idx * 3 + 1
        seed_b = sample_idx * 3 + 2

        # Complex scoring (antithetic: same seed for wt and mut)
        lp_wt_ab = score_structure(model, batch_ab_wt, smask_ab, seed_ab, structure_noise)
        lp_mut_ab = score_structure(model, batch_ab_mut, smask_ab, seed_ab, structure_noise)

        # Chain A scoring
        lp_wt_a = 0.0
        lp_mut_a = 0.0
        if mask_a.any():
            lp_wt_a = score_structure(model, batch_a_wt, smask_a, seed_a, structure_noise)
            lp_mut_a = score_structure(model, batch_a_mut, smask_a, seed_a, structure_noise)

        # Chain B scoring
        lp_wt_b = 0.0
        lp_mut_b = 0.0
        if mask_b.any():
            lp_wt_b = score_structure(model, batch_b_wt, smask_b, seed_b, structure_noise)
            lp_mut_b = score_structure(model, batch_b_mut, smask_b, seed_b, structure_noise)

        # Binding energy proxies
        b_wt = lp_wt_ab - lp_wt_a - lp_wt_b
        b_mut = lp_mut_ab - lp_mut_a - lp_mut_b
        ddg_samples.append(b_mut - b_wt)

    return sum(ddg_samples) / len(ddg_samples)
