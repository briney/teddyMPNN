"""Legacy ↔ current weight conversion for dauparas/ProteinMPNN and dauparas/LigandMPNN.

Legacy checkpoints use different module names, token ordering, and RBF pair
ordering. This module handles all the transformations needed to load legacy
weights into our model and export our weights back to legacy format.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from teddympnn.models.tokens import (
    current_to_legacy_rbf_permutation,
    current_to_legacy_token_permutation,
    expand_pair_permutation,
    legacy_to_current_rbf_permutation,
    legacy_to_current_token_permutation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key renaming maps
# ---------------------------------------------------------------------------

# Legacy key prefix → current key prefix
# Applied left-to-right; first match wins.
_LEGACY_TO_CURRENT_KEY_MAP: list[tuple[str, str]] = [
    # Graph featurization
    (
        "features.embeddings.linear",
        "graph_featurization_module.positional_embedding.embed_positional_features",
    ),
    ("features.edge_embedding", "graph_featurization_module.edge_embedding"),
    ("features.norm_edges", "graph_featurization_module.edge_norm"),
    # LigandMPNN-specific graph featurization
    ("features.node_project_down", "graph_featurization_module.node_embedding"),
    ("features.norm_nodes", "graph_featurization_module.node_norm"),
    ("features.type_linear", "graph_featurization_module.embed_atom_type_features"),
    ("features.y_nodes", "graph_featurization_module.ligand_subgraph_node_embedding"),
    ("features.y_edges", "graph_featurization_module.ligand_subgraph_edge_embedding"),
    ("features.norm_y_nodes", "graph_featurization_module.ligand_subgraph_node_norm"),
    ("features.norm_y_edges", "graph_featurization_module.ligand_subgraph_edge_norm"),
    # LigandMPNN context encoder
    ("W_v", "W_protein_to_ligand_edges_embed"),
    ("W_c", "W_protein_encoding_embed"),
    ("W_nodes_y", "W_ligand_nodes_embed"),
    ("W_edges_y", "W_ligand_edges_embed"),
    ("V_C_norm", "final_context_norm"),
    ("V_C", "W_final_context_embed"),
    ("context_encoder_layers", "protein_ligand_context_encoder_layers"),
    ("y_context_encoder_layers", "ligand_context_encoder_layers"),
]

# Reverse map for export
_CURRENT_TO_LEGACY_KEY_MAP: list[tuple[str, str]] = [
    (current, legacy) for legacy, current in _LEGACY_TO_CURRENT_KEY_MAP
]

# Legacy parameter suffixes differ from PyTorch convention
_LEGACY_SUFFIX_MAP: dict[str, str] = {
    ".w": ".weight",
    ".b": ".bias",
}

_CURRENT_SUFFIX_MAP: dict[str, str] = {
    ".weight": ".w",
    ".bias": ".b",
}


def _rename_key_legacy_to_current(key: str) -> str:
    """Rename a single legacy key to current naming."""
    # Apply suffix mapping
    for old_suffix, new_suffix in _LEGACY_SUFFIX_MAP.items():
        if key.endswith(old_suffix):
            key = key[: -len(old_suffix)] + new_suffix
            break

    # Apply prefix mapping (longest match first to avoid partial matches)
    for legacy_prefix, current_prefix in sorted(
        _LEGACY_TO_CURRENT_KEY_MAP, key=lambda x: -len(x[0])
    ):
        if key.startswith(legacy_prefix):
            return current_prefix + key[len(legacy_prefix) :]

    return key


def _rename_key_current_to_legacy(key: str) -> str:
    """Rename a single current key to legacy naming."""
    # Apply prefix mapping (first match, reversed order for longest-match priority)
    for current_prefix, legacy_prefix in sorted(
        _CURRENT_TO_LEGACY_KEY_MAP, key=lambda x: -len(x[0])
    ):
        if key.startswith(current_prefix):
            key = legacy_prefix + key[len(current_prefix) :]
            break

    # Apply suffix mapping
    for current_suffix, legacy_suffix in _CURRENT_SUFFIX_MAP.items():
        if key.endswith(current_suffix):
            key = key[: -len(current_suffix)] + legacy_suffix
            break

    return key


def _reorder_token_weights(
    state_dict: OrderedDict[str, torch.Tensor],
    permutation: list[int],
) -> None:
    """Reorder W_s and W_out weights along the token dimension in-place."""
    perm = torch.tensor(permutation, dtype=torch.long)

    # W_s.weight: Embedding (vocab_size, hidden_dim) → reorder dim 0
    if "W_s.weight" in state_dict:
        state_dict["W_s.weight"] = state_dict["W_s.weight"][perm]

    # W_out.weight: Linear (hidden_dim → vocab_size) → reorder dim 0
    if "W_out.weight" in state_dict:
        state_dict["W_out.weight"] = state_dict["W_out.weight"][perm]

    # W_out.bias: (vocab_size,) → reorder
    if "W_out.bias" in state_dict:
        state_dict["W_out.bias"] = state_dict["W_out.bias"][perm]


def _reorder_rbf_weights(
    state_dict: OrderedDict[str, torch.Tensor],
    permutation: list[int],
    num_rbf: int = 16,
) -> None:
    """Reorder edge_embedding weights along the RBF pair dimension in-place.

    The edge_embedding weight has shape ``(hidden_dim, 416)``, where the last
    416 dims are ``[16 positional | 400 RBF]``. We only reorder the 400 RBF
    features.
    """
    key = "graph_featurization_module.edge_embedding.weight"
    if key not in state_dict:
        return

    weight = state_dict[key]
    # weight shape: (hidden_dim, 416)
    # Split: positional (16) | RBF (400)
    num_pos = weight.shape[1] - 25 * num_rbf
    pos_part = weight[:, :num_pos]
    rbf_part = weight[:, num_pos:]

    # Expand pair permutation to full RBF feature permutation
    expanded_perm = expand_pair_permutation(permutation, num_rbf)
    perm_tensor = torch.tensor(expanded_perm, dtype=torch.long)

    # Reorder RBF columns
    rbf_part = rbf_part[:, perm_tensor]
    state_dict[key] = torch.cat([pos_part, rbf_part], dim=1)


def _drop_120th_atom_type(
    state_dict: OrderedDict[str, torch.Tensor],
) -> None:
    """Drop the 120th atom type (legacy has 120, current has 119).

    Affects LigandMPNN embedding weights that have atom type as input dim.
    """
    keys_to_check = [
        "graph_featurization_module.embed_atom_type_features.weight",
        "graph_featurization_module.embed_atom_type_features.bias",
        "graph_featurization_module.ligand_subgraph_node_embedding.weight",
    ]
    for key in keys_to_check:
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        # The atom type one-hot occupies the first 120 positions in the input dim
        # We need to drop index 119 (the 120th type, 0-indexed)
        if key.endswith(".bias"):
            continue  # Bias is output-dim, not affected
        if "embed_atom_type_features" in key and key.endswith(".weight"):
            # Linear weight: (out_features, in_features)
            # Input is [120 element + 19 group + 8 period] = 147
            # → [119 element + 19 group + 8 period] = 146
            if tensor.shape[1] == 147:
                state_dict[key] = torch.cat([tensor[:, :119], tensor[:, 120:]], dim=1)
        elif "ligand_subgraph_node_embedding" in key and tensor.shape[1] == 147:
            state_dict[key] = torch.cat([tensor[:, :119], tensor[:, 120:]], dim=1)


def load_legacy_weights(
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load legacy dauparas checkpoint into our model.

    Applies all transformations:
    1. Rename keys (features.* → graph_featurization_module.*, etc.)
    2. Reorder token embeddings in W_s and W_out
    3. Reorder RBF atom pairs in edge_embedding
    4. Drop 120th atom type (LigandMPNN only)
    5. Copy registered buffers from model (not checkpoint)

    Args:
        path: Path to legacy checkpoint file.
        model: Model to load weights into.
        map_location: Device mapping.

    Returns:
        The original (unmodified) checkpoint dict.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Legacy checkpoints store state dict under "model_state_dict"
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Step 1: Rename keys
    renamed = OrderedDict()
    for key, value in raw_state_dict.items():
        new_key = _rename_key_legacy_to_current(key)
        renamed[new_key] = value

    # Step 2: Reorder tokens
    token_perm = legacy_to_current_token_permutation()
    _reorder_token_weights(renamed, token_perm)

    # Step 3: Reorder RBF pairs
    rbf_perm = legacy_to_current_rbf_permutation()
    _reorder_rbf_weights(renamed, rbf_perm)

    # Step 4: Drop 120th atom type (if present)
    _drop_120th_atom_type(renamed)

    # Step 5: Load with strict=False to skip registered buffers,
    # then copy buffers from the model's defaults
    missing, unexpected = model.load_state_dict(renamed, strict=False)

    # Filter out registered buffers from missing keys
    buffer_names = {name for name, _ in model.named_buffers()}
    missing_non_buffer = [k for k in missing if k not in buffer_names]
    if missing_non_buffer:
        logger.warning("Missing non-buffer keys after legacy load: %s", missing_non_buffer)
    if unexpected:
        logger.warning("Unexpected keys in legacy checkpoint: %s", unexpected)

    logger.info("Loaded legacy checkpoint from %s", path)
    return dict(checkpoint)


def convert_to_legacy(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Convert current state_dict to legacy format for export.

    Applies reverse transformations:
    1. Reorder tokens back to 1-letter alphabetical
    2. Reorder RBF pairs back to same-atom-first
    3. Rename keys to legacy naming

    Args:
        state_dict: Current-format state_dict.

    Returns:
        Legacy-format state_dict.
    """
    legacy = OrderedDict(state_dict)

    # Reverse token ordering
    token_perm = current_to_legacy_token_permutation()
    _reorder_token_weights(legacy, token_perm)

    # Reverse RBF ordering
    rbf_perm = current_to_legacy_rbf_permutation()
    _reorder_rbf_weights(legacy, rbf_perm)

    # Rename keys
    final = OrderedDict()
    for key, value in legacy.items():
        new_key = _rename_key_current_to_legacy(key)
        final[new_key] = value

    return final
