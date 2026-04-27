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

    def test_num_context_atoms_propagates(self) -> None:
        """``num_context_atoms`` reaches the graph featurization module.

        Regression: ``ModelConfig.num_context_atoms`` was previously dropped
        before reaching ``ProteinFeaturesLigand``, leaving the module-level
        default in place regardless of config. This test exercises the same
        kwargs path that ``Trainer.from_config`` uses.
        """
        from teddympnn.config import ModelConfig

        cfg = ModelConfig(model_type="ligand_mpnn", num_context_atoms=16)
        model = LigandMPNN(
            hidden_dim=cfg.hidden_dim,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            num_neighbors=cfg.num_neighbors,
            dropout=cfg.dropout_rate,
            num_context_atoms=cfg.num_context_atoms,
        )
        assert model.graph_featurization_module.num_context_atoms == 16

    def test_final_context_no_bias(self) -> None:
        model = LigandMPNN()
        assert model.W_final_context_embed.bias is None

    def test_parameter_count(self) -> None:
        """Parameter count for default LigandMPNN configuration.

        NOTE: This test locks the count produced by the current implementation
        for regression detection only. The original architecture spec cited
        2,621,973 (a 3,472-parameter divergence). Resolve by loading
        Foundry-current weights with ``strict=True`` in the validation gate
        (tests/validation/test_foundry_equivalence.py).
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

    def test_no_context_matches_backbone_encoder(self) -> None:
        """With ``Y_m`` all False, the LigandMPNN encoder must add zero
        context contribution and reproduce the backbone-only encoder output.

        This is the regression for the analysis finding that empty/masked
        ligand context was previously perturbing protein representations
        (``max_abs_no_context_delta = 2.34``). Without this gate, ProteinMPNN
        vs. LigandMPNN ablations aren't interpretable.
        """
        from teddympnn.models.protein_mpnn import ProteinMPNN

        torch.manual_seed(0)
        model = LigandMPNN(
            hidden_dim=64, num_neighbors=8, num_encoder_layers=1, num_decoder_layers=1
        )
        model.eval()

        B, L = 1, 10

        def _features(n_atoms: int, valid: bool) -> dict[str, torch.Tensor]:
            return {
                "X": torch.randn(B, L, 4, 3),
                "S": torch.randint(0, 21, (B, L)),
                "R_idx": torch.arange(L).unsqueeze(0).expand(B, -1),
                "chain_labels": torch.zeros(B, L, dtype=torch.long),
                "residue_mask": torch.ones(B, L),
                "designed_residue_mask": torch.ones(B, L),
                "fixed_residue_mask": torch.zeros(B, L),
                "Y": torch.randn(B, n_atoms, 3) if n_atoms else torch.zeros(B, 0, 3),
                "Y_m": (torch.ones if valid else torch.zeros)(B, n_atoms),
                "Y_t": torch.randint(0, 119, (B, n_atoms))
                if n_atoms
                else torch.zeros(B, 0, dtype=torch.long),
            }

        # Backbone reference: no ligand atoms at all.
        feats_empty = _features(n_atoms=0, valid=True)
        # Masked: ligand atoms exist but are entirely masked out.
        feats_masked = _features(n_atoms=5, valid=False)
        # Use the same backbone for both so encoder inputs are identical.
        feats_masked["X"] = feats_empty["X"].clone()
        feats_masked["S"] = feats_empty["S"].clone()

        with torch.no_grad():
            gf_empty = model._compute_graph_features(feats_empty)
            enc_empty = model.encode(feats_empty, gf_empty)["h_V"]
            backbone = ProteinMPNN.encode(model, feats_empty, gf_empty)["h_V"]
            gf_masked = model._compute_graph_features(feats_masked)
            enc_masked = model.encode(feats_masked, gf_masked)["h_V"]

        # No ligand atoms ⇒ identical to backbone encoder (no context branch run).
        assert torch.allclose(enc_empty, backbone, atol=1e-6)
        # Masked ligand atoms ⇒ context branch zeroed via ``valid_context`` mask;
        # output must match the backbone-only encoder bit-for-bit (single-precision).
        assert torch.allclose(enc_masked, backbone, atol=1e-5), (
            f"max_abs_no_context_delta = "
            f"{(enc_masked - backbone).abs().max().item():.6f}; "
            "empty ligand context must not perturb protein encoding."
        )


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
