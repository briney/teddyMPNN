"""CLI for teddyMPNN."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="teddympnn",
    help="A message passing neural network for protein-protein interfaces.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """teddyMPNN CLI."""
    if version:
        from teddympnn import __version__

        typer.echo(f"teddympnn {__version__}")
        raise typer.Exit()
