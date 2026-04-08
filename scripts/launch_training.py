#!/usr/bin/env python
"""Launch all four training runs sequentially or in parallel.

Usage:
    # Sequential (one GPU, one run at a time)
    python scripts/launch_training.py

    # Specific run only
    python scripts/launch_training.py --run 1

    # Dry run (print commands only)
    python scripts/launch_training.py --dry-run

    # Resume from checkpoint
    python scripts/launch_training.py --run 1 \
        --resume outputs/run1_proteinmpnn_full/checkpoints/step_0050000.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Launch teddyMPNN training runs.")

CONFIGS = {
    1: "configs/run1_proteinmpnn_full.yaml",
    2: "configs/run2_ligandmpnn_full.yaml",
    3: "configs/run3_proteinmpnn_ablation.yaml",
    4: "configs/run4_ligandmpnn_ablation.yaml",
}

DESCRIPTIONS = {
    1: "ProteinMPNN full mix (60/20/20)",
    2: "LigandMPNN full mix (60/20/20)",
    3: "ProteinMPNN ablation (80/0/20, no NVIDIA)",
    4: "LigandMPNN ablation (80/0/20, no NVIDIA)",
}


@app.command()
def main(
    run: Annotated[int | None, typer.Option(help="Run number (1-4). Omit to run all.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Checkpoint to resume from.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print commands only.")] = False,
) -> None:
    """Launch training runs."""
    runs = [run] if run is not None else [1, 2, 3, 4]

    for r in runs:
        if r not in CONFIGS:
            typer.echo(f"Unknown run number: {r}. Valid: 1-4")
            raise typer.Exit(1)

        config_path = CONFIGS[r]
        if not Path(config_path).exists():
            typer.echo(f"Config not found: {config_path}")
            raise typer.Exit(1)

        cmd_parts = ["teddympnn", "train", "--config", config_path]
        if resume is not None:
            cmd_parts.extend(["--resume", str(resume)])

        cmd = " ".join(cmd_parts)
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Run {r}: {DESCRIPTIONS[r]}")
        typer.echo(f"Config: {config_path}")
        typer.echo(f"Command: {cmd}")
        typer.echo(f"{'=' * 60}\n")

        if dry_run:
            continue

        import subprocess

        result = subprocess.run(cmd_parts, check=False)
        if result.returncode != 0:
            typer.echo(f"Run {r} failed with exit code {result.returncode}")
            raise typer.Exit(result.returncode)

        typer.echo(f"Run {r} completed successfully.")


if __name__ == "__main__":
    app()
