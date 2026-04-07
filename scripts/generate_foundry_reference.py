#!/usr/bin/env python
"""Generate Foundry reference tensors for equivalence testing.

Runs inside the brineylab/ligandmpnn Docker container. Uses Foundry's own
forward() method to produce canonical reference outputs for comparison.

Usage (from repo root):
    docker run --rm --gpus all \
        -v $PWD/tests/validation/reference_data:/data \
        --entrypoint python3 brineylab/ligandmpnn:latest \
        /data/generate_foundry_reference.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from mpnn.model.mpnn import LigandMPNN as FoundryLigandMPNN
from mpnn.model.mpnn import ProteinMPNN as FoundryProteinMPNN
from mpnn.utils.weights import load_legacy_weights

DATA_DIR = Path("/data")
WEIGHTS_DIR = DATA_DIR / "weights"
OUTPUT_DIR = DATA_DIR


def _build_network_input(
    B: int,
    L: int,
    seed: int,
    device: str,
    L2: int | None = None,
    N_ligand: int = 0,
) -> dict:
    """Build a ``network_input`` dict suitable for Foundry's forward().

    Args:
        B: Batch size.
        L: Sequence length (single chain), or first-chain length if L2 set.
        seed: Random seed for reproducibility.
        device: Torch device string.
        L2: Optional second-chain length (two-chain complex).
        N_ligand: Number of ligand atoms (0 for ProteinMPNN).
    """
    total_L = L + (L2 or 0)
    gen = torch.Generator(device=device).manual_seed(seed)

    chain = torch.zeros(B, total_L, dtype=torch.long, device=device)
    if L2:
        chain[:, L:] = 1

    if L2:
        R_idx = torch.cat([
            torch.arange(L, device=device),
            torch.arange(L2, device=device),
        ]).unsqueeze(0).expand(B, -1)
        designed = torch.zeros(B, total_L, dtype=torch.bool, device=device)
        designed[:, :L] = True
    else:
        R_idx = torch.arange(total_L, device=device).unsqueeze(0).expand(B, -1)
        designed = torch.ones(B, total_L, dtype=torch.bool, device=device)

    input_features = {
        "X": torch.randn(B, total_L, 37, 3, generator=gen, device=device),
        "X_m": torch.ones(B, total_L, 37, dtype=torch.bool, device=device),
        "S": torch.randint(0, 21, (B, total_L), generator=gen, device=device),
        "R_idx": R_idx,
        "chain_labels": chain,
        "residue_mask": torch.ones(B, total_L, dtype=torch.bool, device=device),
        "designed_residue_mask": designed,
        "structure_noise": 0.0,
        "decode_type": "teacher_forcing",
        "causality_pattern": "auto_regressive",
        "initialize_sequence_embedding_with_ground_truth": True,
        # Defaults required by Foundry's forward()
        "temperature": None,
        "bias": None,
        "pair_bias": None,
        "symmetry_equivalence_group": None,
        "symmetry_weight": None,
        "repeat_sample_num": None,
        "features_to_return": None,
    }

    if N_ligand > 0:
        gen2 = torch.Generator(device=device).manual_seed(seed + 1000)
        input_features["Y"] = torch.randn(B, N_ligand, 3, generator=gen2, device=device)
        input_features["Y_m"] = torch.ones(B, N_ligand, dtype=torch.bool, device=device)
        input_features["Y_t"] = torch.randint(0, 119, (B, N_ligand), generator=gen2, device=device)
        input_features["atomize_side_chains"] = False

    return {"input_features": input_features}


def _to_cpu(d: dict) -> dict:
    """Recursively move all tensors in a dict to CPU."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        elif isinstance(v, dict):
            out[k] = _to_cpu(v)
        else:
            out[k] = v
    return out


def generate_reference(
    model_name: str,
    model: torch.nn.Module,
    cases: dict[str, dict],
    device: str,
) -> None:
    """Run Foundry forward on each test case and save reference data."""
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save the Foundry-converted state dict for weight-loading tests
    sd_path = OUTPUT_DIR / f"{model_name}_foundry_state_dict.pt"
    torch.save({"state_dict": state_dict}, sd_path)
    print(f"  saved state_dict: {sd_path}")

    for case_name, network_input in cases.items():
        print(f"  case: {case_name} ... ", end="", flush=True)
        with torch.no_grad():
            # Foundry's forward returns {"input_features": ..., "graph_features": ...,
            #   "encoder_features": ..., "decoder_features": ...}
            result = model(network_input)

        ref = {
            "state_dict": state_dict,
            "input_features": _to_cpu(result["input_features"]),
            "graph_features": _to_cpu(result["graph_features"]),
            "encoder_features": _to_cpu(result["encoder_features"]),
            "decoder_features": _to_cpu(result["decoder_features"]),
        }
        out_path = OUTPUT_DIR / f"{model_name}_{case_name}.pt"
        torch.save(ref, out_path)
        print(f"saved {out_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- ProteinMPNN ---
    pmpnn_weights = WEIGHTS_DIR / "proteinmpnn_v_48_020.pt"
    if pmpnn_weights.exists():
        print("\n=== ProteinMPNN ===")
        model = FoundryProteinMPNN()
        load_legacy_weights(model, str(pmpnn_weights))
        model = model.to(device).eval()

        cases = {
            "single_chain": _build_network_input(1, 50, seed=42, device=device),
            "two_chain": _build_network_input(1, 30, seed=123, device=device, L2=25),
            "batch": _build_network_input(2, 40, seed=456, device=device),
        }
        generate_reference("proteinmpnn", model, cases, device)
    else:
        print(f"SKIP ProteinMPNN: {pmpnn_weights} not found")

    # --- LigandMPNN ---
    lmpnn_weights = WEIGHTS_DIR / "ligandmpnn_v_32_010_25.pt"
    if lmpnn_weights.exists():
        print("\n=== LigandMPNN ===")
        model = FoundryLigandMPNN()
        load_legacy_weights(model, str(lmpnn_weights))
        model = model.to(device).eval()

        cases = {
            "with_ligand": _build_network_input(
                1, 40, seed=789, device=device, N_ligand=10
            ),
        }
        generate_reference("ligandmpnn", model, cases, device)
    else:
        print(f"SKIP LigandMPNN: {lmpnn_weights} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
