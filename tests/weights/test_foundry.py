"""Tests for Foundry checkpoint loading and export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.weights.foundry import export_foundry_checkpoint, load_foundry_checkpoint


class TestFoundryRoundtrip:
    def test_export_and_reload(self, tmp_path: Path) -> None:
        """Export → reload produces identical state_dict."""
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        export_path = tmp_path / "foundry.pt"
        export_foundry_checkpoint(export_path, model)

        model2 = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        load_foundry_checkpoint(export_path, model2)

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], model2.state_dict()[key]), (
                f"Mismatch for key: {key}"
            )

    def test_checkpoint_has_model_key(self, tmp_path: Path) -> None:
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        path = tmp_path / "foundry.pt"
        export_foundry_checkpoint(path, model)
        ckpt = torch.load(path, weights_only=True)
        assert "model" in ckpt

    def test_config_included(self, tmp_path: Path) -> None:
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        path = tmp_path / "foundry.pt"
        export_foundry_checkpoint(path, model, config={"test": True})
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["config"]["test"] is True
