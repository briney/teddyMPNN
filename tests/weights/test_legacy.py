"""Tests for legacy weight conversion."""

from __future__ import annotations

from collections import OrderedDict

import torch

from teddympnn.models.tokens import (
    TOKEN_ORDER,
    legacy_to_current_token_permutation,
)
from teddympnn.weights.legacy import (
    _drop_120th_atom_type,
    _rename_key_current_to_legacy,
    _rename_key_legacy_to_current,
    _reorder_rbf_weights,
    _reorder_token_weights,
    _restore_120th_atom_type,
    convert_to_legacy,
)


class TestKeyRenaming:
    def test_graph_featurization_keys(self) -> None:
        assert (
            _rename_key_legacy_to_current("features.embeddings.linear.w")
            == "graph_featurization_module.positional_embedding.embed_positional_features.weight"
        )
        assert (
            _rename_key_legacy_to_current("features.edge_embedding.weight")
            == "graph_featurization_module.edge_embedding.weight"
        )
        assert (
            _rename_key_legacy_to_current("features.norm_edges.w")
            == "graph_featurization_module.edge_norm.weight"
        )

    def test_ligand_keys(self) -> None:
        assert _rename_key_legacy_to_current("W_v.w") == "W_protein_to_ligand_edges_embed.weight"
        assert _rename_key_legacy_to_current("W_c.b") == "W_protein_encoding_embed.bias"
        assert _rename_key_legacy_to_current("V_C.weight") == "W_final_context_embed.weight"
        assert _rename_key_legacy_to_current("V_C_norm.w") == "final_context_norm.weight"

    def test_context_encoder_layers(self) -> None:
        key = "context_encoder_layers.0.W1.weight"
        expected = "protein_ligand_context_encoder_layers.0.W1.weight"
        assert _rename_key_legacy_to_current(key) == expected

    def test_unchanged_keys_pass_through(self) -> None:
        assert _rename_key_legacy_to_current("W_e.weight") == "W_e.weight"
        assert _rename_key_legacy_to_current("W_s.weight") == "W_s.weight"
        assert _rename_key_legacy_to_current("W_out.weight") == "W_out.weight"

    def test_roundtrip_key_rename(self) -> None:
        keys = [
            "W_e.weight",
            "W_s.weight",
            "W_out.weight",
            "W_out.bias",
            "encoder_layers.0.W1.weight",
            "decoder_layers.1.dense.W_in.weight",
        ]
        for key in keys:
            legacy = _rename_key_current_to_legacy(key)
            current = _rename_key_legacy_to_current(legacy)
            assert current == key, f"Roundtrip failed for {key}: {legacy} → {current}"


class TestTokenReordering:
    def test_reorder_w_s(self) -> None:
        """Token reordering should permute W_s rows correctly."""
        perm = legacy_to_current_token_permutation()
        state = OrderedDict()
        # Create W_s where each row is identifiable
        state["W_s.weight"] = torch.arange(21 * 4, dtype=torch.float).reshape(21, 4)
        original = state["W_s.weight"].clone()

        _reorder_token_weights(state, perm)

        # After reordering, current index i should have legacy index perm[i]'s data
        for i, tok in enumerate(TOKEN_ORDER):
            legacy_idx = perm[i]
            assert torch.equal(state["W_s.weight"][i], original[legacy_idx]), (
                f"Token {tok}: current[{i}] should equal legacy[{legacy_idx}]"
            )


class TestRBFReordering:
    def test_reorder_preserves_positional(self) -> None:
        """Positional encoding part of edge weights should be unchanged."""
        from teddympnn.models.tokens import legacy_to_current_rbf_permutation

        perm = legacy_to_current_rbf_permutation()
        state = OrderedDict()
        # edge_embedding weight: (hidden_dim, 416) = (128, 16 pos + 400 rbf)
        state["graph_featurization_module.edge_embedding.weight"] = torch.randn(128, 416)
        pos_before = state["graph_featurization_module.edge_embedding.weight"][:, :16].clone()

        _reorder_rbf_weights(state, perm)

        pos_after = state["graph_featurization_module.edge_embedding.weight"][:, :16]
        assert torch.equal(pos_before, pos_after)


class TestConvertToLegacy:
    def test_roundtrip_token_ordering(self) -> None:
        """current → legacy → current should preserve token weights."""

        state = OrderedDict()
        state["W_s.weight"] = torch.randn(21, 128)
        state["W_out.weight"] = torch.randn(21, 128)
        state["W_out.bias"] = torch.randn(21)
        original = OrderedDict({k: v.clone() for k, v in state.items()})

        # Forward: current → legacy
        legacy_state = convert_to_legacy(state)

        # Reverse: legacy → current
        l2c_perm = legacy_to_current_token_permutation()
        _reorder_token_weights(legacy_state, l2c_perm)

        for key in ["W_s.weight", "W_out.weight", "W_out.bias"]:
            # Legacy uses different key names for some keys, but W_s/W_out stay the same
            if key in legacy_state:
                assert torch.allclose(legacy_state[key], original[key], atol=1e-6), (
                    f"Roundtrip failed for {key}"
                )


class TestAtomType120thRoundtrip:
    """Restore↔drop must round-trip the LigandMPNN 120th atom-type slot."""

    def test_restore_inserts_zero_column(self) -> None:
        state = OrderedDict()
        state["graph_featurization_module.embed_atom_type_features.weight"] = torch.randn(
            64, 146
        )
        state[
            "graph_featurization_module.ligand_subgraph_node_embedding.weight"
        ] = torch.randn(128, 146)

        _restore_120th_atom_type(state)

        for key in [
            "graph_featurization_module.embed_atom_type_features.weight",
            "graph_featurization_module.ligand_subgraph_node_embedding.weight",
        ]:
            assert state[key].shape[1] == 147
            # The inserted column at index 119 must be zero.
            assert torch.equal(
                state[key][:, 119], torch.zeros(state[key].shape[0])
            )

    def test_drop_then_restore_is_identity_on_non_119(self) -> None:
        """drop(restore(legacy)) recovers the original input columns
        outside the dropped 119 slot.
        """
        legacy = OrderedDict()
        # Build a legacy 147-wide weight where col 119 is non-zero.
        w = torch.randn(64, 147)
        legacy["graph_featurization_module.embed_atom_type_features.weight"] = w.clone()

        _drop_120th_atom_type(legacy)
        assert (
            legacy["graph_featurization_module.embed_atom_type_features.weight"].shape[1]
            == 146
        )
        _restore_120th_atom_type(legacy)
        restored = legacy["graph_featurization_module.embed_atom_type_features.weight"]
        assert restored.shape == w.shape
        # Columns outside index 119 are preserved.
        assert torch.equal(restored[:, :119], w[:, :119])
        assert torch.equal(restored[:, 120:], w[:, 120:])
        # Restored slot 119 is zero (the slot was discarded during drop).
        assert torch.equal(restored[:, 119], torch.zeros(64))

    def test_convert_to_legacy_restores_atom_type(self) -> None:
        """``convert_to_legacy`` produces a state_dict with the legacy
        147-wide atom-type weights.
        """
        state = OrderedDict()
        state["graph_featurization_module.embed_atom_type_features.weight"] = torch.randn(
            64, 146
        )
        state["graph_featurization_module.embed_atom_type_features.bias"] = torch.randn(64)
        state[
            "graph_featurization_module.ligand_subgraph_node_embedding.weight"
        ] = torch.randn(128, 146)

        legacy = convert_to_legacy(state)
        # Keys are renamed by convert_to_legacy; locate them via the legacy schema.
        type_key_candidates = [k for k in legacy if "type_linear" in k and k.endswith(".w")]
        y_node_key_candidates = [k for k in legacy if "y_nodes" in k and k.endswith(".w")]
        assert type_key_candidates, f"No legacy type_linear weight in {list(legacy)}"
        assert y_node_key_candidates, f"No legacy y_nodes weight in {list(legacy)}"
        assert legacy[type_key_candidates[0]].shape[1] == 147
        assert legacy[y_node_key_candidates[0]].shape[1] == 147
