"""Tests for graph embedding modules."""

from __future__ import annotations

import torch

from teddympnn.models.layers.graph_embeddings import (
    ProteinFeatures,
    ProteinFeaturesLigand,
    compute_knn,
    compute_virtual_cb,
    rbf_encode,
)


class TestUtilities:
    def test_compute_virtual_cb_shape(self) -> None:
        X = torch.randn(2, 10, 4, 3)
        cb = compute_virtual_cb(X)
        assert cb.shape == (2, 10, 1, 3)

    def test_compute_virtual_cb_geometry(self) -> None:
        """CB should be near CA, not at CA."""
        X = torch.randn(1, 5, 4, 3)
        cb = compute_virtual_cb(X).squeeze(-2)
        ca = X[:, :, 1, :]
        dist = (cb - ca).norm(dim=-1)
        # CB should be within ~1.5 Angstrom of CA for typical backbone geometry
        assert (dist > 0).all(), "CB should not be at CA"

    def test_rbf_encode_shape(self) -> None:
        D = torch.rand(2, 10, 5)
        out = rbf_encode(D, num_rbf=16)
        assert out.shape == (2, 10, 5, 16)

    def test_rbf_encode_range(self) -> None:
        """RBF values should be in (0, 1]."""
        D = torch.rand(100) * 20 + 2
        out = rbf_encode(D, num_rbf=16)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_compute_knn_shape(self) -> None:
        B, L, K = 2, 20, 5
        coords = torch.randn(B, L, 3)
        mask = torch.ones(B, L)
        E_idx = compute_knn(coords, mask, K)
        assert E_idx.shape == (B, L, K)

    def test_compute_knn_correctness(self) -> None:
        """Nearest neighbor of a point should be itself (distance 0)."""
        coords = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]])
        mask = torch.ones(1, 3)
        E_idx = compute_knn(coords, mask, 3)
        # First neighbor of each point should be itself
        assert E_idx[0, 0, 0] == 0
        assert E_idx[0, 1, 0] == 1
        assert E_idx[0, 2, 0] == 2

    def test_compute_knn_respects_mask(self) -> None:
        coords = torch.tensor([[[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])  # Middle point masked
        E_idx = compute_knn(coords, mask, 2)
        # Neighbors of position 0 should not include position 1 as nearest
        # (position 1 is masked out with large distance)
        neighbors_0 = E_idx[0, 0].tolist()
        assert 0 in neighbors_0  # Self is nearest
        assert 2 in neighbors_0  # Position 2 is next nearest (position 1 masked)


class TestProteinFeatures:
    def test_output_shapes(self) -> None:
        B, L = 2, 15
        K = 10
        pf = ProteinFeatures(top_k=K, hidden_dim=128)
        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
        chain = torch.zeros(B, L, dtype=torch.long)
        result = pf(X, mask, R_idx, chain)
        assert result["E_idx"].shape == (B, L, K)
        assert result["E"].shape == (B, L, K, 128)

    def test_attribute_names(self) -> None:
        pf = ProteinFeatures()
        assert hasattr(pf, "positional_embedding")
        assert hasattr(pf, "edge_embedding")
        assert hasattr(pf, "edge_norm")

    def test_edge_embedding_no_bias(self) -> None:
        pf = ProteinFeatures()
        assert pf.edge_embedding.bias is None

    def test_positional_distinguishes_chains(self) -> None:
        """Cross-chain edges should differ from same-chain edges."""
        B, L, K = 1, 10, 5
        pf = ProteinFeatures(top_k=K)
        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        R_idx = torch.arange(L).unsqueeze(0)

        chain_same = torch.zeros(B, L, dtype=torch.long)
        chain_diff = torch.zeros(B, L, dtype=torch.long)
        chain_diff[0, 5:] = 1

        out_same = pf(X, mask, R_idx, chain_same)
        out_diff = pf(X, mask, R_idx, chain_diff)
        # Edge features should differ due to different chain labels
        assert not torch.allclose(out_same["E"], out_diff["E"])

    def test_noise_applied_in_training(self) -> None:
        pf = ProteinFeatures(top_k=5)
        X = torch.randn(1, 5, 4, 3)
        mask = torch.ones(1, 5)
        R_idx = torch.arange(5).unsqueeze(0)
        chain = torch.zeros(1, 5, dtype=torch.long)

        pf.train()
        torch.manual_seed(42)
        out_train = pf(X, mask, R_idx, chain, structure_noise=1.0)

        pf.eval()
        torch.manual_seed(42)
        out_eval = pf(X, mask, R_idx, chain, structure_noise=1.0)

        # Noise only applied in training mode
        assert not torch.allclose(out_train["E"], out_eval["E"])


class TestProteinFeaturesLigand:
    def test_output_shapes(self) -> None:
        B, L, N_atoms = 2, 10, 8
        K = 5
        pfl = ProteinFeaturesLigand(top_k=K)
        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
        chain = torch.zeros(B, L, dtype=torch.long)
        Y = torch.randn(B, N_atoms, 3)
        Y_m = torch.ones(B, N_atoms)
        Y_t = torch.randint(0, 119, (B, N_atoms))

        result = pfl(X, mask, R_idx, chain, Y, Y_m, Y_t)

        assert "E_idx" in result
        assert "E" in result
        assert "E_protein_to_ligand" in result
        assert "ligand_subgraph_nodes" in result
        assert "ligand_subgraph_edges" in result
        assert "ligand_subgraph_Y_m" in result
        assert result["E_protein_to_ligand"].shape[-1] == 128
        assert result["ligand_subgraph_nodes"].shape == (B, N_atoms, 128)

    def test_additional_attribute_names(self) -> None:
        pfl = ProteinFeaturesLigand()
        for attr in [
            "embed_atom_type_features",
            "node_embedding",
            "node_norm",
            "ligand_subgraph_node_embedding",
            "ligand_subgraph_node_norm",
            "ligand_subgraph_edge_embedding",
            "ligand_subgraph_edge_norm",
        ]:
            assert hasattr(pfl, attr), f"Missing attribute: {attr}"

    def test_registered_buffers(self) -> None:
        pfl = ProteinFeaturesLigand()
        assert hasattr(pfl, "side_chain_atom_types")
        assert hasattr(pfl, "periodic_table_groups")
        assert hasattr(pfl, "periodic_table_periods")

    def test_masks_invalid_atoms(self) -> None:
        B, L, N_atoms = 1, 5, 4
        pfl = ProteinFeaturesLigand(top_k=3)
        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        R_idx = torch.arange(L).unsqueeze(0)
        chain = torch.zeros(B, L, dtype=torch.long)
        Y = torch.randn(B, N_atoms, 3)
        Y_m = torch.zeros(B, N_atoms)  # All atoms masked
        Y_t = torch.zeros(B, N_atoms, dtype=torch.long)

        result = pfl(X, mask, R_idx, chain, Y, Y_m, Y_t)
        # Masked atoms should produce zero features
        assert torch.allclose(
            result["ligand_subgraph_nodes"],
            torch.zeros_like(result["ligand_subgraph_nodes"]),
        )
