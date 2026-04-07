"""CLI for teddyMPNN."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="teddympnn",
    help="A message passing neural network for protein-protein interfaces.",
    no_args_is_help=True,
)

download_app = typer.Typer(help="Download and prepare datasets.")
app.add_typer(download_app, name="download")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """teddyMPNN CLI."""
    if version:
        from teddympnn import __version__

        typer.echo(f"teddympnn {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Download subcommands
# ---------------------------------------------------------------------------


@download_app.command()
def teddymer(
    output: Annotated[Path, typer.Option(help="Output directory for teddymer data.")] = Path(
        "data/teddymer"
    ),
    workers: Annotated[int, typer.Option(help="Concurrent download workers.")] = 50,
    min_plddt: Annotated[float, typer.Option(help="Min interface pLDDT.")] = 70.0,
    max_pae: Annotated[float, typer.Option(help="Max interface PAE.")] = 10.0,
    min_ifl: Annotated[int, typer.Option(help="Min interface length.")] = 10,
) -> None:
    """Download and preprocess teddymer synthetic dimers."""
    from teddympnn.data.teddymer import (
        download_afdb_structures,
        download_teddymer_metadata,
        filter_teddymer_clusters,
    )

    metadata_dir = download_teddymer_metadata(output / "metadata")
    manifest_path = output / "filtered_manifest.tsv"
    filter_teddymer_clusters(
        metadata_dir,
        manifest_path,
        min_interface_plddt=min_plddt,
        max_interface_pae=max_pae,
        min_interface_length=min_ifl,
    )
    download_afdb_structures(manifest_path, output / "afdb", workers=workers)
    typer.echo(f"Teddymer data prepared in {output}")


@download_app.command("nvidia-complexes")
def nvidia_complexes(
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/nvidia_complexes"),
    min_ipsae: Annotated[float, typer.Option(help="Min ipSAE score.")] = 0.6,
    min_plddt: Annotated[float, typer.Option(help="Min average pLDDT.")] = 70.0,
    max_clashes: Annotated[int, typer.Option(help="Max backbone clashes.")] = 10,
) -> None:
    """Download and filter NVIDIA predicted complexes."""
    from teddympnn.data.nvidia_complexes import (
        download_nvidia_metadata,
        filter_nvidia_metadata,
    )

    csv_path = download_nvidia_metadata(output / "metadata")
    filter_nvidia_metadata(
        csv_path,
        output / "filtered_manifest.tsv",
        min_ipsae=min_ipsae,
        min_plddt=min_plddt,
        max_clashes=max_clashes,
    )
    typer.echo(f"NVIDIA complexes metadata filtered in {output}")


@download_app.command()
def pretrained(
    model: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "ligand_mpnn",
    noise: Annotated[str, typer.Option(help="Noise level (e.g. 020, 010).")] = "020",
    output: Annotated[Path, typer.Option(help="Output directory.")] = Path("weights"),
) -> None:
    """Download pretrained weights from IPD."""
    from teddympnn.weights.io import download_pretrained

    download_pretrained(model, noise, output)
    typer.echo(f"Pretrained weights saved to {output}")
