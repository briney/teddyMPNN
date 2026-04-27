"""Pydantic configuration models for training, evaluation, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataSourceConfig(BaseModel):
    """Configuration for a single training data source."""

    name: str
    weight: float
    path: Path
    source_type: Literal["teddymer", "nvidia", "pdb"]


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_type: Literal["protein_mpnn", "ligand_mpnn"] = "protein_mpnn"
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    num_context_atoms: int = 25
    dropout_rate: float = 0.1


class TrainingConfig(BaseModel):
    """Full training run configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    pretrained_weights: Path
    data_sources: list[DataSourceConfig]
    validation_data_sources: list[DataSourceConfig] = Field(default_factory=list)
    token_budget: int = 10_000
    max_residues: int = 6_000
    min_interface_contacts: int = 4
    learning_rate_factor: float = 2.0
    warmup_steps: int = 4_000
    max_steps: int = 300_000
    grad_clip_max_norm: float | None = None
    structure_noise: float = 0.2
    label_smoothing: float = 0.1
    atomize_partner_sidechains: bool = True
    sidechain_atomization_probability: float = 0.5
    sidechain_atomization_per_residue_probability: float = 0.02
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 8
    seed: int = 42
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 5_000
    save_every_n_steps: int = 10_000
    output_dir: Path = Path("outputs")
    wandb_project: str | None = None

    @model_validator(mode="after")
    def apply_model_defaults(self) -> TrainingConfig:
        """Apply model-specific training defaults when omitted in YAML."""
        if self.model.model_type == "ligand_mpnn":
            if self.model.num_neighbors == 48:
                self.model.num_neighbors = 32
            if self.token_budget == 10_000:
                self.token_budget = 6_000
            if self.structure_noise == 0.2:
                self.structure_noise = 0.1
            if self.grad_clip_max_norm is None:
                self.grad_clip_max_norm = 1.0
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Validated TrainingConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    checkpoint: Path
    data_path: Path
    metrics: list[Literal["recovery", "interface_recovery", "ddg"]]
    num_samples: int = 20
    skempi_path: Path | None = None
