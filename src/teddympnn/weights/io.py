"""Native teddyMPNN checkpoint bundle I/O.

The canonical training artifact is a native checkpoint bundle that stores
model state alongside compatibility metadata needed for export to other
ecosystems (Foundry, legacy dauparas repos).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

FORMAT_VERSION = "teddympnn.v1"

# Pretrained checkpoint URLs from IPD
_IPD_BASE_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"

PRETRAINED_URLS: dict[str, dict[str, str]] = {
    "protein_mpnn": {
        "002": f"{_IPD_BASE_URL}/proteinmpnn_v_48_002.pt",
        "010": f"{_IPD_BASE_URL}/proteinmpnn_v_48_010.pt",
        "020": f"{_IPD_BASE_URL}/proteinmpnn_v_48_020.pt",
        "030": f"{_IPD_BASE_URL}/proteinmpnn_v_48_030.pt",
    },
    "ligand_mpnn": {
        "005": f"{_IPD_BASE_URL}/ligandmpnn_v_32_005_25.pt",
        "010": f"{_IPD_BASE_URL}/ligandmpnn_v_32_010_25.pt",
        "020": f"{_IPD_BASE_URL}/ligandmpnn_v_32_020_25.pt",
        "030": f"{_IPD_BASE_URL}/ligandmpnn_v_32_030_25.pt",
    },
}


def save_checkpoint_bundle(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    step: int = 0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    model_family: str = "protein_mpnn",
    compatibility: dict[str, str | int] | None = None,
) -> None:
    """Save a teddyMPNN-native checkpoint bundle.

    Args:
        path: Output file path.
        model: Model to save.
        optimizer: Optional optimizer state.
        scheduler: Optional scheduler state.
        step: Current training step.
        config: Training configuration dict.
        metrics: Current metrics dict.
        model_family: ``"protein_mpnn"`` or ``"ligand_mpnn"``.
        compatibility: Format compatibility metadata.
    """
    if compatibility is None:
        compatibility = {
            "token_order": "foundry_current",
            "rbf_pair_order": "foundry_current",
            "atom_type_vocabulary": "foundry_current",
            "positional_encoding_classes": 66,
        }

    bundle: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_family": model_family,
        "state_dict": model.state_dict(),
        "step": step,
        "config": config or {},
        "metrics": metrics or {},
        "compatibility": compatibility,
    }
    if optimizer is not None:
        bundle["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        bundle["scheduler"] = scheduler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    logger.info("Saved checkpoint bundle to %s (step %d)", path, step)


def load_checkpoint_bundle(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a teddyMPNN-native checkpoint bundle.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        strict: Whether to require exact key matching.
        map_location: Device mapping for torch.load.

    Returns:
        The full bundle dict (for accessing step, config, metrics, etc.).
    """
    path = Path(path)
    bundle = torch.load(path, map_location=map_location, weights_only=False)

    if bundle.get("format_version") != FORMAT_VERSION:
        logger.warning(
            "Checkpoint format version mismatch: expected %s, got %s",
            FORMAT_VERSION,
            bundle.get("format_version"),
        )

    model.load_state_dict(bundle["state_dict"], strict=strict)
    logger.info("Loaded model state_dict from %s (step %d)", path, bundle.get("step", 0))

    if optimizer is not None and "optimizer" in bundle:
        optimizer.load_state_dict(bundle["optimizer"])
        logger.info("Restored optimizer state")

    if scheduler is not None and "scheduler" in bundle:
        scheduler.load_state_dict(bundle["scheduler"])
        logger.info("Restored scheduler state")

    return bundle


def download_pretrained(
    model_type: str,
    noise_level: str,
    output_dir: str | Path,
) -> Path:
    """Download pretrained weights from IPD servers.

    Args:
        model_type: ``"protein_mpnn"`` or ``"ligand_mpnn"``.
        noise_level: Noise level string (e.g., ``"020"``).
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded checkpoint file.
    """
    if model_type not in PRETRAINED_URLS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(PRETRAINED_URLS)}")

    noise_urls = PRETRAINED_URLS[model_type]
    if noise_level not in noise_urls:
        raise ValueError(
            f"Unknown noise level '{noise_level}' for {model_type}. Choose from {list(noise_urls)}"
        )

    url = noise_urls[noise_level]
    filename = url.split("/")[-1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        logger.info("Pretrained weights already exist at %s", output_path)
        return output_path

    logger.info("Downloading %s to %s", url, output_path)
    torch.hub.download_url_to_file(url, str(output_path))
    logger.info("Download complete: %s", output_path)

    return output_path
