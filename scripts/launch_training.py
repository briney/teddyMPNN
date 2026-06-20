#!/usr/bin/env python
"""Launch the reference training run.

Applies the ``configs/train.yaml`` default (ProteinMPNN, 80/20 teddymer/pdb).

Usage:
    # Run
    python scripts/launch_training.py

    # Print command without executing
    python scripts/launch_training.py --dry-run

    # Resume from checkpoint
    python scripts/launch_training.py \\
        --resume outputs/run1_proteinmpnn_full/checkpoints/step_0050000.pt
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Launch teddyMPNN training runs.")

RUNS: dict[int, tuple[str, str, list[str]]] = {
    1: (
        "ProteinMPNN full mix (80/20)",
        "outputs/run1_proteinmpnn_full",
        [],
    ),
}


@app.command()
def main(
    run: Annotated[int | None, typer.Option(help="Run number (1). Omit to run all.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Checkpoint to resume from.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print commands only.")] = False,
) -> None:
    """Launch training runs."""
    runs = [run] if run is not None else [1]

    for r in runs:
        if r not in RUNS:
            typer.echo(f"Unknown run number: {r}. Valid: 1")
            raise typer.Exit(1)

        description, output_dir, overrides = RUNS[r]
        cmd_parts = ["teddympnn", "train"]
        if resume is not None:
            cmd_parts.extend(["--resume", str(resume)])
        cmd_parts.append(f"output_dir={output_dir}")
        cmd_parts.extend(overrides)

        cmd = " ".join(cmd_parts)
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Run {r}: {description}")
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
