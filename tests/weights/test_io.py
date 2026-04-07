"""Tests for native checkpoint bundle I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.weights.io import (
    FORMAT_VERSION,
    PRETRAINED_URLS,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save → load produces identical state_dict."""
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        ckpt_path = tmp_path / "test.pt"

        save_checkpoint_bundle(
            ckpt_path,
            model,
            step=100,
            config={"hidden_dim": 32},
            metrics={"loss": 1.5},
        )

        model2 = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        bundle = load_checkpoint_bundle(ckpt_path, model2)

        assert bundle["format_version"] == FORMAT_VERSION
        assert bundle["step"] == 100
        assert bundle["config"]["hidden_dim"] == 32
        assert bundle["metrics"]["loss"] == 1.5

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], model2.state_dict()[key]), (
                f"Mismatch for key: {key}"
            )

    def test_saves_optimizer_and_scheduler(self, tmp_path: Path) -> None:
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Take a step so optimizer has state
        dummy = model(
            {
                "X": torch.randn(1, 5, 4, 3),
                "S": torch.randint(0, 21, (1, 5)),
                "R_idx": torch.arange(5).unsqueeze(0),
                "chain_labels": torch.zeros(1, 5, dtype=torch.long),
                "residue_mask": torch.ones(1, 5),
                "designed_residue_mask": torch.ones(1, 5),
            }
        )
        (-dummy["log_probs"].mean()).backward()
        optimizer.step()

        ckpt_path = tmp_path / "with_optim.pt"
        save_checkpoint_bundle(ckpt_path, model, optimizer=optimizer, step=1)

        bundle = load_checkpoint_bundle(ckpt_path, model)
        assert "optimizer" in bundle

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        model = ProteinMPNN(
            hidden_dim=32, num_neighbors=5, num_encoder_layers=1, num_decoder_layers=1
        )
        ckpt_path = tmp_path / "deep" / "nested" / "dir" / "model.pt"
        save_checkpoint_bundle(ckpt_path, model)
        assert ckpt_path.exists()


class TestPretrainedURLs:
    def test_protein_mpnn_urls_exist(self) -> None:
        assert "protein_mpnn" in PRETRAINED_URLS
        assert len(PRETRAINED_URLS["protein_mpnn"]) == 4

    def test_ligand_mpnn_urls_exist(self) -> None:
        assert "ligand_mpnn" in PRETRAINED_URLS
        assert len(PRETRAINED_URLS["ligand_mpnn"]) == 4
