"""Tests for Foundry checkpoint loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import torch

from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.weights.foundry import load_foundry_checkpoint


class TestFoundryLoad:
    def test_load_from_model_key(self, tmp_path: Path) -> None:
        """load_foundry_checkpoint reads weights from the 'model' key."""
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        ckpt_path = tmp_path / "foundry.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        model2 = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        load_foundry_checkpoint(ckpt_path, model2)

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], model2.state_dict()[key]), (
                f"Mismatch for key: {key}"
            )

    def test_returns_checkpoint_dict(self, tmp_path: Path) -> None:
        """load_foundry_checkpoint returns the full checkpoint dict."""
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        ckpt_path = tmp_path / "foundry.pt"
        torch.save({"model": model.state_dict(), "extra": "metadata"}, ckpt_path)

        model2 = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        result = load_foundry_checkpoint(ckpt_path, model2)
        assert "model" in result
        assert result["extra"] == "metadata"
