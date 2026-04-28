"""Pydantic configuration models for training runs and the YAML/CLI loader.

The schema is layered so that ``model_type`` at the top level drives all
model-tuned defaults (architecture hyperparameters, a few training-loop knobs,
and the packaged pretrained-weights path). Fields under ``model:`` and a few
top-level training fields default to ``None``; an ``apply_model_defaults``
validator fills them based on ``model_type``, pulling architecture defaults
from the actual model ``__init__`` signatures so the model classes remain the
single source of truth.

YAML loading and Hydra-style CLI overrides are merged via OmegaConf, then the
resulting plain dict is validated by Pydantic.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from teddympnn.models.ligand_mpnn import LigandMPNN
from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.weights.pretrained import default_pretrained_weights

ModelType = Literal["protein_mpnn", "ligand_mpnn"]
SourceType = Literal["teddymer", "nvidia", "pdb"]

_MODEL_TYPE_TO_CLASS: dict[ModelType, type] = {
    "protein_mpnn": ProteinMPNN,
    "ligand_mpnn": LigandMPNN,
}

_MODEL_TYPE_TRAINING_DEFAULTS: dict[ModelType, dict[str, Any]] = {
    "protein_mpnn": {
        "token_budget": 10_000,
        "structure_noise": 0.20,
        "grad_clip_max_norm": None,
    },
    "ligand_mpnn": {
        "token_budget": 6_000,
        "structure_noise": 0.10,
        "grad_clip_max_norm": 1.0,
    },
}


def _model_init_default(model_cls: type, kwarg: str) -> Any:
    """Return the default value of a kwarg in ``model_cls.__init__``."""
    return inspect.signature(model_cls).parameters[kwarg].default


class DatasetConfig(BaseModel):
    """A single data source: a manifest path and a sampling ratio."""

    path: Path
    ratio: float


class DataConfig(BaseModel):
    """Train and (optional) validation datasets, keyed by source type."""

    train: dict[SourceType, DatasetConfig]
    validation: dict[SourceType, DatasetConfig] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model architecture hyperparameters.

    All fields default to ``None``; the parent ``TrainingConfig`` fills them
    from the chosen ``model_type``'s class defaults via ``apply_model_defaults``.
    """

    hidden_dim: int | None = None
    num_encoder_layers: int | None = None
    num_decoder_layers: int | None = None
    num_neighbors: int | None = None
    dropout: float | None = None
    num_context_atoms: int | None = None  # ligand_mpnn only


class TrainingConfig(BaseModel):
    """Full training run configuration."""

    model_type: ModelType = "protein_mpnn"
    model: ModelConfig = Field(default_factory=ModelConfig)
    pretrained_weights: Path | None = None
    data: DataConfig

    # Model-tuned training knobs — defaulted from model_type when None.
    token_budget: int | None = None
    structure_noise: float | None = None
    grad_clip_max_norm: float | None = None

    # Plain training knobs.
    max_residues: int = 6_000
    min_interface_contacts: int = 4
    learning_rate_factor: float = 2.0
    warmup_steps: int = 4_000
    max_steps: int = 300_000
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
        """Fill model-type-driven defaults for unset fields.

        Architecture defaults come from ``ProteinMPNN.__init__`` /
        ``LigandMPNN.__init__`` signatures so the model classes are the single
        source of truth. The model-tuned training knobs (``token_budget``,
        ``structure_noise``, ``grad_clip_max_norm``) and ``pretrained_weights``
        are also filled from ``model_type`` when omitted.
        """
        model_cls = _MODEL_TYPE_TO_CLASS[self.model_type]
        is_ligand = self.model_type == "ligand_mpnn"

        # num_context_atoms is meaningful only for ligand_mpnn.
        if not is_ligand and self.model.num_context_atoms is not None:
            msg = "num_context_atoms is only valid when model_type='ligand_mpnn'"
            raise ValueError(msg)

        # Architecture defaults from the model class's __init__ signature.
        arch_fields = (
            "hidden_dim",
            "num_encoder_layers",
            "num_decoder_layers",
            "num_neighbors",
            "dropout",
        )
        for field in arch_fields:
            if getattr(self.model, field) is None:
                setattr(self.model, field, _model_init_default(model_cls, field))
        if is_ligand and self.model.num_context_atoms is None:
            self.model.num_context_atoms = _model_init_default(model_cls, "num_context_atoms")

        # Training knob defaults per model_type.
        defaults = _MODEL_TYPE_TRAINING_DEFAULTS[self.model_type]
        if self.token_budget is None:
            self.token_budget = defaults["token_budget"]
        if self.structure_noise is None:
            self.structure_noise = defaults["structure_noise"]
        if self.grad_clip_max_norm is None:
            self.grad_clip_max_norm = defaults["grad_clip_max_norm"]

        # Pretrained-weights default: packaged file for the chosen model_type.
        if self.pretrained_weights is None:
            packaged = default_pretrained_weights(self.model_type)
            if not packaged.exists():
                msg = (
                    f"Default pretrained weights for model_type={self.model_type!r} "
                    f"not found at {packaged}. Provide an explicit "
                    f"pretrained_weights path (CLI: pretrained_weights=<path> or "
                    f"in YAML)."
                )
                raise ValueError(msg)
            self.pretrained_weights = packaged

        return self


def load_training_config(
    yaml_path: Path | str | None,
    overrides: list[str] | None = None,
) -> TrainingConfig:
    """Load a ``TrainingConfig`` from YAML and/or Hydra-style CLI overrides.

    Args:
        yaml_path: Path to a base YAML config, or ``None`` to start from an
            empty config (overrides supply everything).
        overrides: Hydra-style dotlist overrides, e.g.
            ``["model.hidden_dim=256", "data.train.teddymer.ratio=0.5"]``.

    Returns:
        Fully validated ``TrainingConfig`` with model-typed defaults applied.
    """
    base = OmegaConf.load(yaml_path) if yaml_path is not None else OmegaConf.create({})
    cli = OmegaConf.from_dotlist(list(overrides or []))
    merged = OmegaConf.merge(base, cli)
    plain = OmegaConf.to_container(merged, resolve=True)
    return TrainingConfig.model_validate(plain or {})


__all__ = [
    "DataConfig",
    "DatasetConfig",
    "ModelConfig",
    "ModelType",
    "SourceType",
    "TrainingConfig",
    "load_training_config",
]
