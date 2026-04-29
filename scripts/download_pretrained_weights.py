#!/usr/bin/env python
"""Download the bundled ProteinMPNN/LigandMPNN base checkpoints.

Run once after cloning the repo, before committing. The downloaded files are
then version-controlled and shipped in the wheel so a fresh ``pip install``
yields a runnable training default.

Usage:
    python scripts/download_pretrained_weights.py
    python scripts/download_pretrained_weights.py --force
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from teddympnn.weights.io import PRETRAINED_URLS, download_pretrained
from teddympnn.weights.pretrained import PACKAGED_WEIGHTS

PRETRAINED_DIR = Path(__file__).resolve().parents[1] / "src/teddympnn/weights/pretrained"

app = typer.Typer(help="Download bundled pretrained MPNN weights.")
console = Console()


def _noise_level_for(model_type: str, filename: str) -> str:
    """Reverse-lookup the noise level that maps to ``filename`` for ``model_type``."""
    for noise, url in PRETRAINED_URLS[model_type].items():
        if url.endswith(filename):
            return noise
    msg = f"No PRETRAINED_URLS entry for {model_type} matching filename {filename!r}"
    raise ValueError(msg)


@app.command()
def main(
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even if file already exists.")
    ] = False,
) -> None:
    """Download every checkpoint declared in ``PACKAGED_WEIGHTS``."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Target dir:[/bold] {PRETRAINED_DIR}\n")

    for model_type, filename in PACKAGED_WEIGHTS.items():
        dest = PRETRAINED_DIR / filename
        if dest.exists() and force:
            dest.unlink()

        noise = _noise_level_for(model_type, filename)
        console.print(f"[cyan]{model_type}[/cyan] ({noise}) → {filename}")
        download_pretrained(model_type, noise, PRETRAINED_DIR)

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
