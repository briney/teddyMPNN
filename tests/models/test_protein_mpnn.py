"""Tests for the ProteinMPNN model."""

from __future__ import annotations

import torch

from teddympnn.models.protein_mpnn import ProteinMPNN


def _make_input_features(
    B: int = 2,
    L: int = 20,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create synthetic input features for testing."""
    return {
        "X": torch.randn(B, L, 4, 3, device=device),
        "S": torch.randint(0, 21, (B, L), device=device),
        "R_idx": torch.arange(L, device=device).unsqueeze(0).expand(B, -1),
        "chain_labels": torch.zeros(B, L, dtype=torch.long, device=device),
        "residue_mask": torch.ones(B, L, device=device),
        "designed_residue_mask": torch.ones(B, L, device=device),
        "fixed_residue_mask": torch.zeros(B, L, device=device),
    }


class TestProteinMPNNForward:
    def test_output_shape(self) -> None:
        model = ProteinMPNN(
            hidden_dim=64, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        features = _make_input_features(B=1, L=10)
        output = model(features)
        assert "log_probs" in output
        assert output["log_probs"].shape == (1, 10, 21)

    def test_log_probs_valid(self) -> None:
        """Log probs should be <= 0 and exp should sum to ~1."""
        model = ProteinMPNN(
            hidden_dim=64, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        model.eval()
        features = _make_input_features(B=1, L=8)
        with torch.no_grad():
            output = model(features)
        log_probs = output["log_probs"]
        assert (log_probs <= 0).all()
        probs = torch.exp(log_probs)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


class TestProteinMPNNArchitecture:
    def test_module_attribute_names(self) -> None:
        model = ProteinMPNN()
        assert hasattr(model, "graph_featurization_module")
        assert hasattr(model, "W_e")
        assert hasattr(model, "W_s")
        assert hasattr(model, "encoder_layers")
        assert hasattr(model, "decoder_layers")
        assert hasattr(model, "W_out")

    def test_default_layer_counts(self) -> None:
        model = ProteinMPNN()
        assert len(model.encoder_layers) == 3
        assert len(model.decoder_layers) == 3

    def test_default_neighbors(self) -> None:
        model = ProteinMPNN()
        assert model.num_neighbors == 48

    def test_parameter_count(self) -> None:
        """Parameter count for default ProteinMPNN configuration.

        NOTE: exact match against Foundry checkpoints is verified in the
        validation gate (tests/validation/). This test locks the count for
        regression detection.
        """
        model = ProteinMPNN()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 1_660_485, f"Expected 1,660,485 params, got {total_params}"


class TestProteinMPNNEncode:
    def test_encode_output_shapes(self) -> None:
        K = 10
        model = ProteinMPNN(
            hidden_dim=64, num_neighbors=K, num_encoder_layers=1, num_decoder_layers=1
        )
        features = _make_input_features(B=1, L=15)
        graph = model._compute_graph_features(features)
        enc = model.encode(features, graph)
        assert enc["h_V"].shape == (1, 15, 64)
        assert enc["h_E"].shape == (1, 15, K, 64)


class TestProteinMPNNScore:
    def test_score_shape(self) -> None:
        model = ProteinMPNN(
            hidden_dim=64, num_neighbors=10, num_encoder_layers=1, num_decoder_layers=1
        )
        features = _make_input_features(B=1, L=8)
        scores = model.score(features)
        assert scores.shape == (1, 8)
        assert (scores <= 0).all()


class TestProteinMPNNGradient:
    def test_gradient_flows(self) -> None:
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        model.train()
        features = _make_input_features(B=1, L=6)
        output = model(features)
        loss = -output["log_probs"].mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
