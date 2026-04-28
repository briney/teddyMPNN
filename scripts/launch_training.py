#!/usr/bin/env python
"""Launch the four reference training runs sequentially.

Each run is a thin Hydra-style override of ``configs/train.yaml``:

    Run 1: ProteinMPNN, 60/20/20 teddymer/nvidia/pdb (the train.yaml default)
    Run 2: LigandMPNN, 60/20/20
    Run 3: ProteinMPNN ablation (no NVIDIA, 80/20 teddymer/pdb)
    Run 4: LigandMPNN ablation (no NVIDIA, 80/20)

Ablations zero out the NVIDIA sampling ratio, which prevents the source from
being drawn but still parses its manifest at startup. If you want to actually
skip the NVIDIA dataset entirely, copy ``configs/train.yaml`` to a one-off file
and remove the NVIDIA entries.

Usage:
    # Run all four
    python scripts/launch_training.py

    # One run only
    python scripts/launch_training.py --run 1

    # Print commands without executing
    python scripts/launch_training.py --dry-run

    # Resume from checkpoint
    python scripts/launch_training.py --run 1 \\
        --resume outputs/run1_proteinmpnn_full/checkpoints/step_0050000.pt
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(help="Launch teddyMPNN training runs.")

_ABLATION_RATIOS = [
    "data.train.teddymer.ratio=0.80",
    "data.train.nvidia.ratio=0.0",
    "data.train.pdb.ratio=0.20",
    "data.validation.teddymer.ratio=0.80",
    "data.validation.nvidia.ratio=0.0",
    "data.validation.pdb.ratio=0.20",
]

RUNS: dict[int, tuple[str, str, list[str]]] = {
    1: (
        "ProteinMPNN full mix (60/20/20)",
        "outputs/run1_proteinmpnn_full",
        [],
    ),
    2: (
        "LigandMPNN full mix (60/20/20)",
        "outputs/run2_ligandmpnn_full",
        ["model_type=ligand_mpnn"],
    ),
    3: (
        "ProteinMPNN ablation (no NVIDIA, 80/20)",
        "outputs/run3_proteinmpnn_ablation",
        list(_ABLATION_RATIOS),
    ),
    4: (
        "LigandMPNN ablation (no NVIDIA, 80/20)",
        "outputs/run4_ligandmpnn_ablation",
        ["model_type=ligand_mpnn", *_ABLATION_RATIOS],
    ),
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
        if r not in RUNS:
            typer.echo(f"Unknown run number: {r}. Valid: 1-4")
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
