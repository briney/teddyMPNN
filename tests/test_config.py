"""Tests for the Pydantic configuration models and the YAML/CLI loader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from teddympnn.config import (
    DataConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    load_training_config,
)


def _minimal_yaml(tmp_path: Path, **extra: Any) -> Path:
    """Write a minimal valid YAML with only the required ``data`` field."""
    data: dict[str, Any] = {
        "data": {
            "train": {
                "pdb": {"path": str(tmp_path / "train.tsv"), "ratio": 1.0},
            },
        },
        "pretrained_weights": str(tmp_path / "dummy.pt"),
    }
    data.update(extra)
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


class TestModelTypeDefaults:
    def test_protein_mpnn_arch_defaults(self, tmp_path: Path) -> None:
        """With model_type=protein_mpnn and no model overrides, defaults match ProteinMPNN."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "protein_mpnn",
                "pretrained_weights": str(tmp_path / "dummy.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
            },
        )
        assert cfg.model.hidden_dim == 128
        assert cfg.model.num_encoder_layers == 3
        assert cfg.model.num_decoder_layers == 3
        assert cfg.model.num_neighbors == 48
        assert cfg.model.dropout == 0.1
        assert cfg.model.num_context_atoms is None  # not applicable to protein_mpnn

    def test_ligand_mpnn_arch_defaults(self, tmp_path: Path) -> None:
        """With model_type=ligand_mpnn, defaults pull from LigandMPNN.__init__."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "ligand_mpnn",
                "pretrained_weights": str(tmp_path / "dummy.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
            },
        )
        assert cfg.model.num_neighbors == 32
        assert cfg.model.num_context_atoms == 25

    def test_protein_mpnn_training_defaults(self, tmp_path: Path) -> None:
        """ProteinMPNN training-knob defaults: token_budget=10_000, structure_noise=0.20."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "protein_mpnn",
                "pretrained_weights": str(tmp_path / "dummy.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
            },
        )
        assert cfg.token_budget == 10_000
        assert cfg.structure_noise == 0.20
        assert cfg.grad_clip_max_norm is None

    def test_ligand_mpnn_training_defaults(self, tmp_path: Path) -> None:
        """LigandMPNN training-knob defaults: token_budget=6_000, structure_noise=0.10, clip=1.0."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "ligand_mpnn",
                "pretrained_weights": str(tmp_path / "dummy.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
            },
        )
        assert cfg.token_budget == 6_000
        assert cfg.structure_noise == 0.10
        assert cfg.grad_clip_max_norm == 1.0

    def test_user_overrides_win(self, tmp_path: Path) -> None:
        """User-provided values for model fields and training knobs are respected."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "protein_mpnn",
                "model": {"hidden_dim": 256, "num_neighbors": 64},
                "pretrained_weights": str(tmp_path / "dummy.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
                "token_budget": 12_345,
            },
        )
        assert cfg.model.hidden_dim == 256
        assert cfg.model.num_neighbors == 64
        # un-overridden field still defaulted
        assert cfg.model.num_encoder_layers == 3
        assert cfg.token_budget == 12_345

    def test_num_context_atoms_rejected_for_protein_mpnn(self, tmp_path: Path) -> None:
        """Setting num_context_atoms with model_type=protein_mpnn raises."""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(
                {
                    "model_type": "protein_mpnn",
                    "model": {"num_context_atoms": 25},
                    "pretrained_weights": str(tmp_path / "dummy.pt"),
                    "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
                },
            )

    def test_invalid_model_type(self, tmp_path: Path) -> None:
        """Unknown model_type raises ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(
                {
                    "model_type": "bogus",
                    "pretrained_weights": str(tmp_path / "dummy.pt"),
                    "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
                },
            )


class TestPretrainedWeightsResolution:
    def test_explicit_path_skips_existence_check(self, tmp_path: Path) -> None:
        """An explicit pretrained_weights path is accepted even if the file is missing."""
        cfg = TrainingConfig.model_validate(
            {
                "model_type": "protein_mpnn",
                "pretrained_weights": str(tmp_path / "missing.pt"),
                "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
            },
        )
        assert cfg.pretrained_weights == tmp_path / "missing.pt"

    def test_unset_resolves_to_packaged(self, tmp_path: Path) -> None:
        """Unset pretrained_weights resolves via default_pretrained_weights when the file exists."""
        fake_packaged = tmp_path / "fake_packaged.pt"
        fake_packaged.write_bytes(b"")  # exists, content irrelevant for validator
        with patch(
            "teddympnn.config.default_pretrained_weights",
            return_value=fake_packaged,
        ):
            cfg = TrainingConfig.model_validate(
                {
                    "model_type": "protein_mpnn",
                    "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
                },
            )
        assert cfg.pretrained_weights == fake_packaged

    def test_unset_with_missing_packaged_raises(self, tmp_path: Path) -> None:
        """If the packaged default file is missing, ValidationError is raised."""
        missing = tmp_path / "not_there.pt"
        with (
            patch("teddympnn.config.default_pretrained_weights", return_value=missing),
            pytest.raises(ValidationError, match="Default pretrained weights"),
        ):
            TrainingConfig.model_validate(
                {
                    "model_type": "protein_mpnn",
                    "data": {"train": {"pdb": {"path": str(tmp_path / "t.tsv"), "ratio": 1.0}}},
                },
            )


class TestDataConfig:
    def test_round_trip(self, tmp_path: Path) -> None:
        """DataConfig parses train/validation dicts keyed by source type."""
        data = DataConfig.model_validate(
            {
                "train": {
                    "teddymer": {"path": str(tmp_path / "tr_t.tsv"), "ratio": 0.6},
                    "pdb": {"path": str(tmp_path / "tr_p.tsv"), "ratio": 0.4},
                },
                "validation": {
                    "pdb": {"path": str(tmp_path / "v_p.tsv"), "ratio": 1.0},
                },
            },
        )
        assert set(data.train.keys()) == {"teddymer", "pdb"}
        assert data.train["teddymer"].ratio == 0.6
        assert data.validation["pdb"].path == tmp_path / "v_p.tsv"

    def test_invalid_source_key(self, tmp_path: Path) -> None:
        """Source keys other than the configured Literal are rejected."""
        with pytest.raises(ValidationError):
            DataConfig.model_validate(
                {"train": {"unknown": {"path": str(tmp_path / "x.tsv"), "ratio": 1.0}}},
            )

    def test_validation_optional(self, tmp_path: Path) -> None:
        """validation defaults to an empty dict."""
        data = DataConfig.model_validate(
            {"train": {"pdb": {"path": str(tmp_path / "p.tsv"), "ratio": 1.0}}},
        )
        assert data.validation == {}


class TestLoadTrainingConfig:
    def test_yaml_only(self, tmp_path: Path) -> None:
        """Loader reads the YAML and applies model-type defaults."""
        path = _minimal_yaml(tmp_path, model_type="protein_mpnn")
        cfg = load_training_config(path, [])
        assert cfg.model_type == "protein_mpnn"
        assert cfg.model.hidden_dim == 128
        assert cfg.token_budget == 10_000

    def test_overrides_merged(self, tmp_path: Path) -> None:
        """Hydra-style dotlist overrides override YAML values."""
        path = _minimal_yaml(tmp_path, model_type="protein_mpnn")
        cfg = load_training_config(
            path,
            [
                "model.hidden_dim=256",
                f"data.train.pdb.path={tmp_path}/override.tsv",
                "data.train.pdb.ratio=0.5",
                "max_steps=42",
            ],
        )
        assert cfg.model.hidden_dim == 256
        assert cfg.data.train["pdb"].path == tmp_path / "override.tsv"
        assert cfg.data.train["pdb"].ratio == 0.5
        assert cfg.max_steps == 42

    def test_override_switches_model_type(self, tmp_path: Path) -> None:
        """Switching model_type via override applies the new model's defaults."""
        path = _minimal_yaml(tmp_path, model_type="protein_mpnn")
        cfg = load_training_config(path, ["model_type=ligand_mpnn"])
        assert cfg.model_type == "ligand_mpnn"
        assert cfg.model.num_neighbors == 32
        assert cfg.token_budget == 6_000

    def test_no_yaml(self, tmp_path: Path) -> None:
        """With no YAML path, overrides alone can produce a valid config."""
        cfg = load_training_config(
            None,
            [
                "model_type=protein_mpnn",
                f"pretrained_weights={tmp_path}/dummy.pt",
                f"data.train.pdb.path={tmp_path}/t.tsv",
                "data.train.pdb.ratio=1.0",
            ],
        )
        assert cfg.model_type == "protein_mpnn"


class TestPlainModels:
    def test_dataset_config_round_trip(self, tmp_path: Path) -> None:
        ds = DatasetConfig(path=tmp_path / "x.tsv", ratio=0.5)
        assert ds.path == tmp_path / "x.tsv"
        assert ds.ratio == 0.5

    def test_model_config_defaults_unset(self) -> None:
        """ModelConfig fields default to None until apply_model_defaults runs."""
        cfg = ModelConfig()
        assert cfg.hidden_dim is None
        assert cfg.num_neighbors is None
        assert cfg.num_context_atoms is None
