"""Fixtures and utilities for Foundry equivalence validation tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

REFERENCE_DIR = Path(__file__).parent / "reference_data"


def _has_reference_files() -> bool:
    """Check if reference .pt files exist (not just the directory)."""
    return REFERENCE_DIR.exists() and any(
        f.name.startswith("proteinmpnn_") and f.suffix == ".pt"
        for f in REFERENCE_DIR.iterdir()
        if f.name != "generate_foundry_reference.py"
    )


requires_reference_data = pytest.mark.skipif(
    not _has_reference_files(),
    reason=(
        "Foundry reference data not found. "
        "Run scripts/generate_foundry_reference.py in the Foundry container."
    ),
)


def load_reference(name: str) -> dict[str, Any]:
    """Load a reference .pt file by name (without directory or extension)."""
    path = REFERENCE_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"Reference file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
