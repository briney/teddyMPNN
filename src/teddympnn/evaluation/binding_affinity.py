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
import re
from pathlib import Path  # noqa: TC003 — used at runtime in predict_ddg signature
from typing import Any

import torch

from teddympnn.evaluation._batch import (
    build_eval_batch,
    extract_chain_view,
    load_eval_features,
)
from teddympnn.models.ligand_mpnn import LigandMPNN
from teddympnn.models.tokens import (
    AMINO_ACIDS_1TO3,
    idx_to_token,
    token_to_idx,
)

logger = logging.getLogger(__name__)


# Splits a mutation body like "52a" or "-3" into (resnum, icode).
_MUT_BODY_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_type(model: torch.nn.Module) -> str:
    return "ligand_mpnn" if isinstance(model, LigandMPNN) else "protein_mpnn"


def _parse_mutation_body(body: str) -> tuple[int, str]:
    """Split a mutation residue body like ``"52a"`` into ``(52, "a")``."""
    m = _MUT_BODY_RE.match(body)
    if m is None:
        msg = f"Cannot parse mutation residue identifier: '{body}'"
        raise ValueError(msg)
    return int(m.group(1)), m.group(2)


def _apply_mutations(
    features: dict[str, Any],
    chain_ids: list[str],
    residue_numbers: list[int],
    residue_icodes: list[str],
    mutations: dict[str, dict[str, str | None]],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply mutations to the sequence tensor.

    Args:
        features: Unbatched feature dict.
        chain_ids: Per-residue chain IDs.
        residue_numbers: Per-residue PDB residue numbers.
        residue_icodes: Per-residue PDB insertion codes (``""`` if none).
        mutations: ``{chain_id: {mutation_str: None}}`` where mutation_str
            is ``"A45G"`` or ``"L52aG"`` (wt + resnum [+ icode] + mut).

    Returns:
        Tuple of (mutant features, mutation_mask as ``(L,)`` bool tensor).
    """
    S_mut = features["S"].clone()
    mutation_mask = torch.zeros(len(S_mut), dtype=torch.bool)

    for chain_id, chain_muts in mutations.items():
        for mut_str in chain_muts:
            wt_aa_1 = mut_str[0]
            mut_aa_1 = mut_str[-1]
            resnum, icode = _parse_mutation_body(mut_str[1:-1])

            wt_aa_3 = AMINO_ACIDS_1TO3.get(wt_aa_1)
            mut_aa_3 = AMINO_ACIDS_1TO3.get(mut_aa_1)
            if wt_aa_3 is None or mut_aa_3 is None:
                msg = f"Unknown amino acid in mutation '{mut_str}'"
                raise ValueError(msg)

            found = False
            for i, (cid, rn, ic) in enumerate(
                zip(chain_ids, residue_numbers, residue_icodes, strict=True)
            ):
                if cid == chain_id and rn == resnum and ic == icode:
                    expected_idx = token_to_idx[wt_aa_3]
                    actual_idx = features["S"][i].item()
                    if actual_idx != expected_idx:
                        actual_name = idx_to_token.get(actual_idx, "???")
                        logger.warning(
                            "Mutation %s on chain %s: expected %s at %d%s but found %s",
                            mut_str,
                            chain_id,
                            wt_aa_3,
                            resnum,
                            icode,
                            actual_name,
                        )
                    S_mut[i] = token_to_idx[mut_aa_3]
                    mutation_mask[i] = True
                    found = True
                    break

            if not found:
                msg = f"Residue {chain_id}:{resnum}{icode} not found in structure"
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
        input_features: Batched feature dict (B=1) — should already include
            ligand context tensors when ``model`` is a ``LigandMPNN``.
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
    per_residue: torch.Tensor = model.score(features, score_mask=score_mask)
    return float(per_residue.sum().item())


@torch.no_grad()
def score_complex(
    model: torch.nn.Module,
    structure_path: str | Path,
    *,
    designed_chains: list[str] | None = None,
    num_samples: int = 20,
    structure_noise: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Score a complex by averaging per-residue log-probabilities.

    Loads a structure file, builds the appropriate ProteinMPNN or LigandMPNN
    batch, and runs ``num_samples`` teacher-forced scoring passes with
    independent decoding orders. Returns the mean per-residue
    log-probability at every designed position.

    Args:
        model: ProteinMPNN or LigandMPNN model.
        structure_path: Path to a PDB or mmCIF file.
        designed_chains: Chain IDs whose residues are scored. Defaults to
            all chains in the structure (i.e., score every residue).
        num_samples: Number of scoring passes to average (each with a
            fresh decoding order).
        structure_noise: Backbone coordinate noise std (A) for ensemble
            scoring.
        device: Device for computation. Defaults to the model's device.

    Returns:
        ``(L_designed,)`` float tensor of mean log-probabilities at the
        designed residues, in structure order.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    model_type = _model_type(model)
    features = load_eval_features(structure_path, model_type=model_type)

    chain_ids: list[str] = features["chain_ids"]
    L = len(chain_ids)

    if designed_chains is None:
        designed_set: set[str] = set(chain_ids)
    else:
        designed_set = set(designed_chains)

    designed_mask = torch.tensor([cid in designed_set for cid in chain_ids], dtype=torch.bool)
    if not designed_mask.any():
        msg = (
            f"No residues match designed_chains={sorted(designed_set)}; "
            f"available chains: {sorted(set(chain_ids))}"
        )
        raise ValueError(msg)

    fixed_mask = (~designed_mask) & torch.ones(L, dtype=torch.bool)
    batch = build_eval_batch(
        features,
        designed_mask,
        device,
        model_type=model_type,
        fixed_residue_mask=fixed_mask,
    )
    score_mask = designed_mask.unsqueeze(0).to(device)

    accum = torch.zeros(L, dtype=torch.float32, device=device)
    for sample_idx in range(num_samples):
        torch.manual_seed(sample_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed(sample_idx)
        sample_features = batch
        if structure_noise > 0:
            noise = torch.randn_like(batch["X"])
            sample_features = {**batch, "X": batch["X"] + structure_noise * noise}
        per_residue: torch.Tensor = model.score(sample_features, score_mask=score_mask)
        accum = accum + per_residue.squeeze(0)

    mean_log_probs = accum / max(num_samples, 1)
    designed_indices = designed_mask.nonzero(as_tuple=True)[0].to(device)
    return mean_log_probs.index_select(0, designed_indices).cpu()


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
            mutation_str is formatted as ``"A45G"`` or ``"L52aG"``
            (wt + resnum [+ insertion code] + mut).
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

    model_type = _model_type(model)
    features = load_eval_features(structure_path, model_type=model_type)
    chain_ids: list[str] = features["chain_ids"]
    residue_numbers: list[int] = features["residue_numbers"]
    residue_icodes: list[str] = features.get("residue_icodes", [""] * len(chain_ids))

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
        residue_icodes,
        mutations,
    )

    # Extract chain-level views
    chain_a_wt, chain_a_ids, _, _ = extract_chain_view(
        features,
        chain_ids,
        partner_a,
        residue_numbers=residue_numbers,
        residue_icodes=residue_icodes,
    )
    chain_a_mut, _, _, _ = extract_chain_view(
        mut_features,
        chain_ids,
        partner_a,
        residue_numbers=residue_numbers,
        residue_icodes=residue_icodes,
    )
    chain_b_wt, chain_b_ids, _, _ = extract_chain_view(
        features,
        chain_ids,
        partner_b,
        residue_numbers=residue_numbers,
        residue_icodes=residue_icodes,
    )
    chain_b_mut, _, _, _ = extract_chain_view(
        mut_features,
        chain_ids,
        partner_b,
        residue_numbers=residue_numbers,
        residue_icodes=residue_icodes,
    )

    # Build mutation masks for each view
    mask_ab = mutation_mask  # (L_AB,)
    mask_a = _map_mask_to_chain(mutation_mask, chain_ids, partner_a)  # (L_A,)
    mask_b = _map_mask_to_chain(mutation_mask, chain_ids, partner_b)  # (L_B,)

    # Build batches (B=1) — ddG decomposition treats every residue as
    # designed, so the fixed_residue_mask is empty here.
    L_ab = len(chain_ids)
    L_a = len(chain_a_ids)
    L_b = len(chain_b_ids)
    full_designed_ab = torch.ones(L_ab, dtype=torch.bool)
    full_designed_a = torch.ones(L_a, dtype=torch.bool)
    full_designed_b = torch.ones(L_b, dtype=torch.bool)
    empty_fixed_ab = torch.zeros(L_ab, dtype=torch.bool)
    empty_fixed_a = torch.zeros(L_a, dtype=torch.bool)
    empty_fixed_b = torch.zeros(L_b, dtype=torch.bool)

    batch_ab_wt = build_eval_batch(
        features,
        full_designed_ab,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_ab,
    )
    batch_ab_mut = build_eval_batch(
        mut_features,
        full_designed_ab,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_ab,
    )
    batch_a_wt = build_eval_batch(
        chain_a_wt,
        full_designed_a,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_a,
    )
    batch_a_mut = build_eval_batch(
        chain_a_mut,
        full_designed_a,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_a,
    )
    batch_b_wt = build_eval_batch(
        chain_b_wt,
        full_designed_b,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_b,
    )
    batch_b_mut = build_eval_batch(
        chain_b_mut,
        full_designed_b,
        device,
        model_type=model_type,
        fixed_residue_mask=empty_fixed_b,
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
