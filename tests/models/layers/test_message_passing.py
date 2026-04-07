"""Tests for message passing layers and helper functions."""

from __future__ import annotations

import torch

from teddympnn.models.layers.message_passing import (
    DecLayer,
    EncLayer,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)


class TestGatherFunctions:
    def test_gather_nodes_shape(self) -> None:
        B, L, K, H = 2, 10, 5, 32
        node_features = torch.randn(B, L, H)
        neighbor_idx = torch.randint(0, L, (B, L, K))
        out = gather_nodes(node_features, neighbor_idx)
        assert out.shape == (B, L, K, H)

    def test_gather_nodes_correctness(self) -> None:
        L, H = 4, 3
        node_features = torch.arange(L * H, dtype=torch.float).reshape(1, L, H)
        neighbor_idx = torch.tensor([[[1, 3], [0, 2], [3, 1], [0, 0]]])
        out = gather_nodes(node_features, neighbor_idx)
        # Position 0, neighbor 0 should be node 1's features
        assert torch.allclose(out[0, 0, 0], node_features[0, 1])
        # Position 0, neighbor 1 should be node 3's features
        assert torch.allclose(out[0, 0, 1], node_features[0, 3])

    def test_gather_edges_passthrough(self) -> None:
        """When edge features are already (B, L, K, H), return as-is."""
        B, L, K, H = 2, 10, 5, 32
        edge_features = torch.randn(B, L, K, H)
        neighbor_idx = torch.randint(0, L, (B, L, K))
        out = gather_edges(edge_features, neighbor_idx)
        assert torch.equal(out, edge_features)

    def test_cat_neighbors_nodes_shape(self) -> None:
        B, L, K = 2, 10, 5
        H_node, H_edge = 32, 64
        h_nodes = torch.randn(B, L, H_node)
        h_edges = torch.randn(B, L, K, H_edge)
        E_idx = torch.randint(0, L, (B, L, K))
        out = cat_neighbors_nodes(h_nodes, h_edges, E_idx)
        assert out.shape == (B, L, K, H_edge + H_node)


class TestEncLayer:
    def test_output_shapes(self) -> None:
        B, L, K, H = 2, 20, 10, 128
        layer = EncLayer(H, 3 * H)
        h_V = torch.randn(B, L, H)
        h_E = torch.randn(B, L, K, H)
        E_idx = torch.randint(0, L, (B, L, K))
        mask_V = torch.ones(B, L)
        mask_E = torch.ones(B, L, K)
        h_V_out, h_E_out = layer(h_V, h_E, E_idx, mask_V, mask_E)
        assert h_V_out.shape == (B, L, H)
        assert h_E_out.shape == (B, L, K, H)

    def test_mask_zeros_output(self) -> None:
        B, L, K, H = 1, 5, 3, 32
        layer = EncLayer(H, 3 * H)
        layer.eval()
        h_V = torch.randn(B, L, H)
        h_E = torch.randn(B, L, K, H)
        E_idx = torch.randint(0, L, (B, L, K))
        mask_V = torch.zeros(B, L)
        mask_E = torch.zeros(B, L, K)
        h_V_out, h_E_out = layer(h_V, h_E, E_idx, mask_V, mask_E)
        assert torch.allclose(h_V_out, torch.zeros_like(h_V_out))
        assert torch.allclose(h_E_out, torch.zeros_like(h_E_out))

    def test_attribute_names(self) -> None:
        layer = EncLayer(128, 384)
        for attr in [
            "W1",
            "W2",
            "W3",
            "W11",
            "W12",
            "W13",
            "norm1",
            "norm2",
            "norm3",
            "dropout1",
            "dropout2",
            "dropout3",
            "dense",
            "act",
        ]:
            assert hasattr(layer, attr), f"Missing attribute: {attr}"

    def test_gradient_flow(self) -> None:
        H = 32
        layer = EncLayer(H, 3 * H)
        h_V = torch.randn(1, 5, H, requires_grad=True)
        h_E = torch.randn(1, 5, 3, H, requires_grad=True)
        E_idx = torch.randint(0, 5, (1, 5, 3))
        mask_V = torch.ones(1, 5)
        mask_E = torch.ones(1, 5, 3)
        h_V_out, h_E_out = layer(h_V, h_E, E_idx, mask_V, mask_E)
        (h_V_out.sum() + h_E_out.sum()).backward()
        assert h_V.grad is not None
        assert h_E.grad is not None
        for param in layer.parameters():
            assert param.grad is not None


class TestDecLayer:
    def test_output_shape(self) -> None:
        B, L, K, H = 2, 20, 10, 128
        # ProteinMPNN decoder: num_in = 4H, caller passes 3H, layer adds H
        layer = DecLayer(H, 4 * H)
        h_V = torch.randn(B, L, H)
        h_E = torch.randn(B, L, K, 3 * H)  # num_in - H
        mask_V = torch.ones(B, L)
        mask_E = torch.ones(B, L, K)
        h_V_out = layer(h_V, h_E, mask_V, mask_E)
        assert h_V_out.shape == (B, L, H)

    def test_variable_num_in(self) -> None:
        """DecLayer supports different num_in for different contexts.

        Caller provides num_in - H features; layer prepends source H.
        """
        H = 64
        for num_in in [2 * H, 3 * H, 4 * H]:
            layer = DecLayer(H, num_in)
            h_V = torch.randn(1, 5, H)
            h_E = torch.randn(1, 5, 3, num_in - H)  # caller passes num_in - H
            mask_V = torch.ones(1, 5)
            mask_E = torch.ones(1, 5, 3)
            out = layer(h_V, h_E, mask_V, mask_E)
            assert out.shape == (1, 5, H)

    def test_attribute_names(self) -> None:
        layer = DecLayer(128, 512)
        for attr in ["W1", "W2", "W3", "norm1", "norm2", "dropout1", "dropout2", "dense", "act"]:
            assert hasattr(layer, attr), f"Missing attribute: {attr}"

    def test_gradient_flow(self) -> None:
        H = 32
        layer = DecLayer(H, 4 * H)
        h_V = torch.randn(1, 5, H, requires_grad=True)
        h_E = torch.randn(1, 5, 3, 3 * H, requires_grad=True)  # num_in - H
        mask_V = torch.ones(1, 5)
        mask_E = torch.ones(1, 5, 3)
        out = layer(h_V, h_E, mask_V, mask_E)
        out.sum().backward()
        assert h_V.grad is not None
        for param in layer.parameters():
            assert param.grad is not None
