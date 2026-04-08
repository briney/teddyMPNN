"""Tests for Pydantic configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from teddympnn.config import DataSourceConfig, EvalConfig, ModelConfig, TrainingConfig


class TestModelConfig:
    def test_defaults_protein_mpnn(self) -> None:
        """Default ModelConfig matches ProteinMPNN defaults."""
        cfg = ModelConfig()
        assert cfg.model_type == "protein_mpnn"
        assert cfg.hidden_dim == 128
        assert cfg.num_encoder_layers == 3
        assert cfg.num_decoder_layers == 3
        assert cfg.num_neighbors == 48
        assert cfg.dropout_rate == 0.1

    def test_ligand_mpnn_overrides(self) -> None:
        """LigandMPNN typical config with reduced neighbors."""
        cfg = ModelConfig(model_type="ligand_mpnn", num_neighbors=32)
        assert cfg.model_type == "ligand_mpnn"
        assert cfg.num_neighbors == 32

    def test_invalid_model_type(self) -> None:
        """Invalid model type raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(model_type="invalid")


class TestTrainingConfig:
    def test_from_yaml(self, tmp_path: Path) -> None:
        """TrainingConfig loads from a valid YAML file."""
        config_data = {
            "pretrained_weights": "weights/proteinmpnn_v_48_020.pt",
            "data_sources": [
                {
                    "name": "teddymer",
                    "weight": 0.6,
                    "path": "data/teddymer",
                    "source_type": "teddymer",
                },
            ],
            "max_steps": 1000,
            "seed": 123,
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = TrainingConfig.from_yaml(yaml_path)
        assert cfg.max_steps == 1000
        assert cfg.seed == 123
        assert len(cfg.data_sources) == 1
        assert cfg.data_sources[0].name == "teddymer"

    def test_defaults(self) -> None:
        """TrainingConfig has sensible defaults."""
        cfg = TrainingConfig(
            pretrained_weights=Path("weights/test.pt"),
            data_sources=[
                DataSourceConfig(name="test", weight=1.0, path=Path("data/test"), source_type="pdb")
            ],
        )
        assert cfg.token_budget == 10_000
        assert cfg.warmup_steps == 4_000
        assert cfg.max_steps == 300_000
        assert cfg.label_smoothing == 0.1
        assert cfg.mixed_precision is True
        assert cfg.gradient_checkpointing is True
        assert cfg.output_dir == Path("outputs")

    def test_missing_required_fields(self) -> None:
        """Missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            TrainingConfig()  # type: ignore[call-arg]

    def test_invalid_data_source_type(self) -> None:
        """Invalid source_type raises ValidationError."""
        with pytest.raises(ValidationError):
            DataSourceConfig(name="bad", weight=1.0, path=Path("data"), source_type="unknown")

    def test_model_config_nested(self) -> None:
        """Model config can be overridden in YAML."""
        cfg = TrainingConfig(
            pretrained_weights=Path("weights/test.pt"),
            data_sources=[
                DataSourceConfig(name="test", weight=1.0, path=Path("data/test"), source_type="pdb")
            ],
            model=ModelConfig(model_type="ligand_mpnn", num_neighbors=32),
        )
        assert cfg.model.model_type == "ligand_mpnn"
        assert cfg.model.num_neighbors == 32


class TestEvalConfig:
    def test_valid_config(self) -> None:
        """EvalConfig with valid fields."""
        cfg = EvalConfig(
            checkpoint=Path("outputs/step_1000.pt"),
            data_path=Path("data/test"),
            metrics=["recovery", "interface_recovery"],
        )
        assert cfg.num_samples == 20
        assert cfg.skempi_path is None

    def test_invalid_metric(self) -> None:
        """Invalid metric name raises ValidationError."""
        with pytest.raises(ValidationError):
            EvalConfig(
                checkpoint=Path("test.pt"),
                data_path=Path("data"),
                metrics=["invalid_metric"],
            )
