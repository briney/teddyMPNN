"""Tests for the LigandMPNN model."""

from __future__ import annotations

import torch

from teddympnn.models.ligand_mpnn import LigandMPNN


def _make_ligand_input_features(
    B: int = 2,
    L: int = 15,
    N_atoms: int = 10,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create synthetic input features with ligand atoms."""
    return {
        "X": torch.randn(B, L, 4, 3, device=device),
        "S": torch.randint(0, 21, (B, L), device=device),
        "R_idx": torch.arange(L, device=device).unsqueeze(0).expand(B, -1),
        "chain_labels": torch.zeros(B, L, dtype=torch.long, device=device),
        "residue_mask": torch.ones(B, L, device=device),
        "designed_residue_mask": torch.ones(B, L, device=device),
        "fixed_residue_mask": torch.zeros(B, L, device=device),
        "Y": torch.randn(B, N_atoms, 3, device=device),
        "Y_m": torch.ones(B, N_atoms, device=device),
        "Y_t": torch.randint(0, 119, (B, N_atoms), device=device),
    }


class TestLigandMPNNForward:
    def test_output_shape(self) -> None:
        model = LigandMPNN(
            hidden_dim=64, num_neighbors=8, num_encoder_layers=1, num_decoder_layers=1
        )
        features = _make_ligand_input_features(B=1, L=10, N_atoms=5)
        output = model(features)
        assert output["log_probs"].shape == (1, 10, 21)

    def test_log_probs_valid(self) -> None:
        model = LigandMPNN(
            hidden_dim=64, num_neighbors=8, num_encoder_layers=1, num_decoder_layers=1
        )
        model.eval()
        features = _make_ligand_input_features(B=1, L=8, N_atoms=4)
        with torch.no_grad():
            output = model(features)
        log_probs = output["log_probs"]
        assert (log_probs <= 0).all()
        probs = torch.exp(log_probs)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


class TestLigandMPNNArchitecture:
    def test_additional_attributes(self) -> None:
        model = LigandMPNN()
        for attr in [
            "W_protein_to_ligand_edges_embed",
            "W_protein_encoding_embed",
            "W_ligand_nodes_embed",
            "W_ligand_edges_embed",
            "W_final_context_embed",
            "final_context_norm",
            "protein_ligand_context_encoder_layers",
            "ligand_context_encoder_layers",
        ]:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

    def test_context_encoder_layer_counts(self) -> None:
        model = LigandMPNN()
        assert len(model.protein_ligand_context_encoder_layers) == 2
        assert len(model.ligand_context_encoder_layers) == 2

    def test_default_neighbors(self) -> None:
        """LigandMPNN uses k=32, not k=48."""
        model = LigandMPNN()
        assert model.num_neighbors == 32

    def test_final_context_no_bias(self) -> None:
        model = LigandMPNN()
        assert model.W_final_context_embed.bias is None

    def test_parameter_count(self) -> None:
        """Parameter count for default LigandMPNN configuration.

        NOTE: exact match against Foundry checkpoints is verified in the
        validation gate (tests/validation/). This test locks the count for
        regression detection.
        """
        model = LigandMPNN()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 2_618_501, f"Expected 2,618,501 params, got {total_params}"


class TestLigandMPNNContext:
    def test_context_modifies_encoding(self) -> None:
        """Ligand context should change encoder output vs no context."""
        model = LigandMPNN(
            hidden_dim=64, num_neighbors=8, num_encoder_layers=1, num_decoder_layers=1
        )
        model.eval()

        features_with = _make_ligand_input_features(B=1, L=8, N_atoms=5)
        features_without = _make_ligand_input_features(B=1, L=8, N_atoms=5)
        # Zero out ligand atoms
        features_without["Y_m"] = torch.zeros_like(features_without["Y_m"])
        # Use same backbone
        features_without["X"] = features_with["X"].clone()

        with torch.no_grad():
            graph_with = model._compute_graph_features(features_with)
            enc_with = model.encode(features_with, graph_with)
            graph_without = model._compute_graph_features(features_without)
            enc_without = model.encode(features_without, graph_without)

        # Context should make a non-zero contribution
        assert not torch.allclose(enc_with["h_V"], enc_without["h_V"], atol=1e-4)


class TestLigandMPNNGradient:
    def test_gradient_flows(self) -> None:
        model = LigandMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        model.train()
        features = _make_ligand_input_features(B=1, L=6, N_atoms=3)
        output = model(features)
        loss = -output["log_probs"].mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
