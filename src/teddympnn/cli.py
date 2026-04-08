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

checkpoints_app = typer.Typer(help="Checkpoint management.")
app.add_typer(checkpoints_app, name="checkpoints")

evaluate_app = typer.Typer(help="Evaluate a trained model.")
app.add_typer(evaluate_app, name="evaluate")


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


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="Path to training config YAML.")] = ...,
    resume: Annotated[Path | None, typer.Option(help="Checkpoint to resume from.")] = None,
) -> None:
    """Train a teddyMPNN model from a YAML config."""
    from teddympnn.config import TrainingConfig
    from teddympnn.training.trainer import Trainer

    training_config = TrainingConfig.from_yaml(config)
    trainer = Trainer.from_config(training_config)

    if resume is not None:
        trainer.load_checkpoint(resume)

    trainer.train()
    typer.echo("Training complete.")


# ---------------------------------------------------------------------------
# Checkpoint subcommands
# ---------------------------------------------------------------------------


@checkpoints_app.command("export-foundry")
def export_foundry(
    checkpoint: Annotated[Path, typer.Option(help="teddyMPNN checkpoint path.")] = ...,
    output: Annotated[Path, typer.Option(help="Output Foundry checkpoint path.")] = ...,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
) -> None:
    """Export a teddyMPNN checkpoint to Foundry format."""

    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.foundry import export_foundry_checkpoint
    from teddympnn.weights.io import load_checkpoint_bundle

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()

    bundle = load_checkpoint_bundle(checkpoint, model, map_location="cpu")
    export_foundry_checkpoint(output, model, config=bundle.get("config"))
    typer.echo(f"Exported Foundry checkpoint to {output}")


# ---------------------------------------------------------------------------
# Evaluate subcommands
# ---------------------------------------------------------------------------


@evaluate_app.command()
def recovery(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,
    data: Annotated[Path, typer.Option(help="Test data manifest path.")] = ...,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
) -> None:
    """Evaluate interface sequence recovery on a test set."""
    import torch

    from teddympnn.data.collator import PaddingCollator
    from teddympnn.data.dataset import PPIDataset
    from teddympnn.data.sampler import TokenBudgetBatchSampler
    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.io import load_checkpoint_bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_checkpoint_bundle(checkpoint, model, map_location=device)
    model = model.to(device)
    model.eval()

    dataset = PPIDataset(data, include_ligand_atoms=(model_type == "ligand_mpnn"))
    collator = PaddingCollator()
    sampler = TokenBudgetBatchSampler(dataset.lengths, token_budget=10_000, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    total_correct = 0
    total_designed = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            output = model(batch)
            preds = output["log_probs"].argmax(dim=-1)
            mask = batch["designed_residue_mask"].bool()
            total_correct += ((preds == batch["S"]) & mask).sum().item()
            total_designed += mask.sum().item()

    rec = total_correct / max(total_designed, 1)
    typer.echo(f"Sequence recovery: {rec:.4f} ({total_correct}/{total_designed})")


@evaluate_app.command()
def ddg(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,
    skempi: Annotated[Path, typer.Option(help="SKEMPI data directory.")] = ...,
    num_samples: Annotated[int, typer.Option(help="Monte Carlo samples.")] = 20,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
) -> None:
    """Evaluate binding affinity prediction on SKEMPI v2.0."""
    typer.echo(
        "ddG evaluation requires the evaluation module (Phase 4). "
        "Use `teddympnn evaluate recovery` for sequence recovery."
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Score command
# ---------------------------------------------------------------------------


@app.command()
def score(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,
    pdb: Annotated[Path, typer.Option(help="PDB/mmCIF structure file.")] = ...,
    chains: Annotated[str, typer.Option(help="Design chain IDs (comma-separated).")] = ...,
    num_samples: Annotated[int, typer.Option(help="Monte Carlo samples.")] = 1,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
) -> None:
    """Score a structure with a trained model."""
    import torch

    from teddympnn.data.features import derive_backbone, parse_structure
    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.io import load_checkpoint_bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_checkpoint_bundle(checkpoint, model, map_location=device)
    model = model.to(device)

    features = parse_structure(pdb)
    X, X_m = derive_backbone(features["xyz_37"], features["xyz_37_m"])
    features["X"] = X
    features["X_m"] = X_m

    design_chain_set = set(chains.split(","))
    chain_ids: list[str] = features["chain_ids"]
    L = len(chain_ids)
    designed = torch.zeros(L, dtype=torch.bool)
    for i, cid in enumerate(chain_ids):
        if cid in design_chain_set:
            designed[i] = True

    # Build single-example batch
    batch = {
        "X": features["X"].unsqueeze(0).to(device),
        "S": features["S"].unsqueeze(0).to(device),
        "R_idx": features["R_idx"].unsqueeze(0).to(device),
        "chain_labels": features["chain_labels"].unsqueeze(0).to(device),
        "residue_mask": features["residue_mask"].unsqueeze(0).to(device),
        "designed_residue_mask": designed.unsqueeze(0).to(device),
        "fixed_residue_mask": (~designed).unsqueeze(0).to(device),
    }

    scores = []
    for _ in range(num_samples):
        per_residue = model.score(batch, score_mask=designed.unsqueeze(0).to(device))
        total = per_residue.sum().item()
        scores.append(total)

    import statistics

    mean_score = statistics.mean(scores)
    typer.echo(f"Log-likelihood score: {mean_score:.4f} (n={num_samples})")
