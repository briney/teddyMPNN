"""Packaged pretrained model weights.

Looks up packaged ``.pt`` files shipped inside the wheel via importlib.resources
so the right base weights are available immediately after ``pip install``.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from typing import Final

PACKAGED_WEIGHTS: Final[dict[str, str]] = {
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
}


def default_pretrained_weights(model_type: str) -> Path:
    """Return the packaged default weights path for ``model_type``.

    Args:
        model_type: ``"protein_mpnn"`` or ``"ligand_mpnn"``.

    Returns:
        Filesystem path to the packaged ``.pt`` file. The file is not guaranteed
        to exist — callers should verify and surface a clear error if missing
        (e.g., when running from a source checkout before the binary weights
        have been committed).

    Raises:
        KeyError: If ``model_type`` has no packaged default.
    """
    filename = PACKAGED_WEIGHTS[model_type]
    resource = files("teddympnn.weights.pretrained").joinpath(filename)
    with as_file(resource) as path:
        return Path(path)
