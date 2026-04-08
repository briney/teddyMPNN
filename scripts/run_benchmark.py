#!/usr/bin/env python
"""Run benchmarks comparing fine-tuned teddyMPNN models against baselines.

This script is a convenience wrapper around ``teddympnn evaluate benchmark``
that auto-discovers checkpoints from the standard output directory layout.

Usage:
    # Auto-discover all trained models and run benchmarks
    python scripts/run_benchmark.py

    # Specify checkpoint paths manually
    python scripts/run_benchmark.py \
        --checkpoint outputs/run1_proteinmpnn_full/checkpoints/step_0300000.pt \
        --checkpoint outputs/run2_ligandmpnn_full/checkpoints/step_0300000.pt

    # Custom output
    python scripts/run_benchmark.py --output results/benchmark.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(help="Run teddyMPNN benchmarks.")
console = Console()

# Standard output directories from configs/
_RUN_DIRS = {
    "teddyMPNN-P (full)": ("outputs/run1_proteinmpnn_full", "protein_mpnn"),
    "teddyMPNN-L (full)": ("outputs/run2_ligandmpnn_full", "ligand_mpnn"),
    "teddyMPNN-P (ablation)": ("outputs/run3_proteinmpnn_ablation", "protein_mpnn"),
    "teddyMPNN-L (ablation)": ("outputs/run4_ligandmpnn_ablation", "ligand_mpnn"),
}

_BASELINES = {
    "ProteinMPNN (vanilla)": ("weights/v_48_020.pt", "protein_mpnn"),
    "LigandMPNN (vanilla)": ("weights/v_32_010_25.pt", "ligand_mpnn"),
}


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest checkpoint in a run's checkpoint directory."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


@app.command()
def main(
    checkpoint: Annotated[list[Path] | None, typer.Option(help="Checkpoint paths.")] = None,
    val_manifest: Annotated[Path, typer.Option(help="Validation manifest.")] = Path(
        "data/manifests/val_manifest.tsv"
    ),
    skempi_dir: Annotated[Path, typer.Option(help="SKEMPI data directory.")] = Path("data/skempi"),
    num_samples: Annotated[int, typer.Option(help="Monte Carlo samples for ddG.")] = 20,
    output: Annotated[Path, typer.Option(help="Output JSON path.")] = Path(
        "results/benchmark.json"
    ),
    skip_recovery: Annotated[bool, typer.Option(help="Skip sequence recovery.")] = False,
    skip_skempi: Annotated[bool, typer.Option(help="Skip SKEMPI evaluation.")] = False,
) -> None:
    """Run benchmarks and print comparison tables."""
    from teddympnn.evaluation.benchmark import (
        ModelSpec,
        print_comparison_table,
        run_benchmark,
    )

    models: list[ModelSpec] = []

    # Add baselines
    for name, (path, model_type) in _BASELINES.items():
        if Path(path).exists():
            models.append(ModelSpec(name=name, checkpoint=Path(path), model_type=model_type))
            console.print(f"  [dim]Baseline:[/dim] {name} ({path})")
        else:
            console.print(f"  [yellow]Skipping baseline {name} (not found: {path})[/yellow]")

    # Add fine-tuned models
    if checkpoint:
        for i, ckpt in enumerate(checkpoint):
            # Infer model type from path
            model_type = "ligand_mpnn" if "ligand" in str(ckpt).lower() else "protein_mpnn"
            models.append(ModelSpec(name=f"checkpoint_{i}", checkpoint=ckpt, model_type=model_type))
    else:
        # Auto-discover from standard directories
        for name, (run_dir, model_type) in _RUN_DIRS.items():
            ckpt = _find_latest_checkpoint(Path(run_dir))
            if ckpt is not None:
                models.append(ModelSpec(name=name, checkpoint=ckpt, model_type=model_type))
                console.print(f"  [dim]Fine-tuned:[/dim] {name} ({ckpt})")
            else:
                console.print(f"  [yellow]Skipping {name} (no checkpoint in {run_dir})[/yellow]")

    if not models:
        console.print("[red]No models found to benchmark.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Benchmarking {len(models)} models[/bold]\n")

    test_manifests = None
    if not skip_recovery and val_manifest.exists():
        test_manifests = {"held_out": val_manifest}

    skempi = skempi_dir if not skip_skempi else None

    report = run_benchmark(
        models,
        test_manifests=test_manifests,
        skempi_dir=skempi,
        num_samples=num_samples,
    )

    print_comparison_table(report)
    report.save_json(output)
    console.print(f"\n[green]Report saved to {output}[/green]")


if __name__ == "__main__":
    app()
