#!/usr/bin/env python
"""End-to-end data preparation pipeline.

Downloads teddymer and PDB data sources, applies quality filters, and prepares
unified train/val manifests with reproducible 95/5 splits.

Usage:
    python scripts/prepare_data.py --output-dir data

    # Skip PDB
    python scripts/prepare_data.py --output-dir data --skip-pdb

    # Dry run (show what would be done)
    python scripts/prepare_data.py --output-dir data --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(help="Prepare teddyMPNN training data.")
console = Console()


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option(help="Root output directory.")] = Path("data"),
    skip_pdb: Annotated[bool, typer.Option(help="Skip PDB complexes.")] = False,
    val_fraction: Annotated[float, typer.Option(help="Validation fraction.")] = 0.05,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    workers: Annotated[int, typer.Option(help="Download concurrency.")] = 50,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print only.")] = False,
) -> None:
    """Download data, filter, and prepare manifests."""
    teddymer_dir = output_dir / "teddymer"
    pdb_dir = output_dir / "pdb"
    manifest_dir = output_dir / "manifests"

    steps = [
        ("Download teddymer metadata + structures", teddymer_dir),
    ]
    if not skip_pdb:
        steps.append(("Query and download PDB complexes", pdb_dir))
    steps.append(("Prepare unified train/val manifests", manifest_dir))

    console.print("\n[bold]Data preparation pipeline[/bold]")
    for i, (desc, path) in enumerate(steps, 1):
        console.print(f"  {i}. {desc} → {path}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — no actions taken.[/yellow]")
        return

    # Step 1: Teddymer
    console.rule("Teddymer")
    from teddympnn.data.teddymer import TeddymerPrepareConfig, prepare_teddymer_data

    teddymer_result = prepare_teddymer_data(
        TeddymerPrepareConfig(output_dir=teddymer_dir, workers=workers)
    )
    teddymer_manifest = teddymer_result.all_manifest_path

    # Step 2: PDB
    pdb_manifest = None
    if not skip_pdb:
        console.rule("PDB Complexes")
        from teddympnn.data.pdb_complexes import (
            download_pdb_structures,
            query_pdb_complexes,
        )

        pdb_list = pdb_dir / "pdb_ids.txt"
        query_pdb_complexes(pdb_list)
        download_pdb_structures(pdb_list, pdb_dir / "structures")
        pdb_manifest = pdb_dir / "structures" / "manifest.tsv"

    # Step 3: Unified manifests
    console.rule("Unified Manifests")
    from teddympnn.data.splits import prepare_manifests

    train_path, val_path = prepare_manifests(
        manifest_dir,
        teddymer_manifest=teddymer_manifest,
        pdb_manifest=pdb_manifest,
        val_fraction=val_fraction,
        seed=seed,
    )

    console.print()
    console.print(f"[green]Train manifest:[/green] {train_path}")
    console.print(f"[green]Val manifest:[/green]   {val_path}")
    console.print("[bold green]Data preparation complete.[/bold green]")


if __name__ == "__main__":
    app()
