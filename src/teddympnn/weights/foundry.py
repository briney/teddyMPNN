"""Foundry-format checkpoint loading and export.

Foundry's "current" format stores checkpoints with the same module hierarchy
we use, so loading is direct. Exporting produces a checkpoint that Foundry
users can load via ``model.load_state_dict(checkpoint["model"])``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def load_foundry_checkpoint(
    path: str | Path,
    model: nn.Module,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a Foundry-current format checkpoint.

    Foundry checkpoints store the state dict under the ``"model"`` key.

    Args:
        path: Path to the Foundry checkpoint file.
        model: Model to load weights into.
        strict: Whether to require exact key matching.
        map_location: Device mapping.

    Returns:
        The full checkpoint dict.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning("Missing keys in Foundry checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in Foundry checkpoint: %s", unexpected)

    logger.info("Loaded Foundry checkpoint from %s", path)
    return dict(checkpoint)


def export_foundry_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any] | None = None,
) -> None:
    """Export a Foundry-compatible checkpoint from a teddyMPNN model.

    Writes a checkpoint that Foundry users can load directly:
    ``model.load_state_dict(torch.load(path)["model"])``.

    Args:
        path: Output file path.
        model: Model whose state_dict to export.
        config: Optional config metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
    }
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    logger.info("Exported Foundry checkpoint to %s", path)
