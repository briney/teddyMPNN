"""Tests for legacy weight loading."""

from __future__ import annotations

from collections import OrderedDict

import torch

from teddympnn.models.tokens import (
    TOKEN_ORDER,
    legacy_to_current_token_permutation,
)
from teddympnn.weights.legacy import (
    _drop_120th_atom_type,
    _rename_key_legacy_to_current,
    _reorder_rbf_weights,
    _reorder_token_weights,
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

    def test_unchanged_keys_pass_through(self) -> None:
        assert _rename_key_legacy_to_current("W_e.weight") == "W_e.weight"
        assert _rename_key_legacy_to_current("W_s.weight") == "W_s.weight"
        assert _rename_key_legacy_to_current("W_out.weight") == "W_out.weight"


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


class TestDropAtomType:
    """_drop_120th_atom_type reduces input dim from 147 to 146."""

    def test_drop_reduces_dim(self) -> None:
        state = OrderedDict()
        state["graph_featurization_module.embed_atom_type_features.weight"] = torch.randn(64, 147)

        _drop_120th_atom_type(state)

        assert state["graph_featurization_module.embed_atom_type_features.weight"].shape[1] == 146

    def test_drop_preserves_columns_outside_119(self) -> None:
        state = OrderedDict()
        w = torch.randn(64, 147)
        state["graph_featurization_module.embed_atom_type_features.weight"] = w.clone()

        _drop_120th_atom_type(state)

        result = state["graph_featurization_module.embed_atom_type_features.weight"]
        assert torch.equal(result[:, :119], w[:, :119])
        assert torch.equal(result[:, 119:], w[:, 120:])
