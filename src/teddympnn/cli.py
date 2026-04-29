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
    workers: Annotated[int, typer.Option(help="Process pool size for dimer assembly.")] = 16,
    chunk_size: Annotated[int, typer.Option(help="Manifest rows per FoldSeek batch.")] = 50_000,
    foldseek_threads: Annotated[
        int | None,
        typer.Option(help="--threads for foldseek convert2pdb (default: all cores)."),
    ] = None,
    foldseek_db: Annotated[
        Path | None,
        typer.Option(help="Override path to the FoldSeek DB prefix (no suffix)."),
    ] = None,
    min_plddt: Annotated[float, typer.Option(help="Min interface pLDDT.")] = 70.0,
    max_pae: Annotated[float, typer.Option(help="Max interface PAE.")] = 10.0,
    min_ifl: Annotated[int, typer.Option(help="Min interface length.")] = 10,
) -> None:
    """Download and preprocess teddymer synthetic dimers."""
    from teddympnn.data.teddymer import (
        download_teddymer_metadata,
        extract_and_assemble_dimers,
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
    db_path = foldseek_db or (
        metadata_dir / "teddymer" / "dir_ted_afdb50_cath_dimerdb" / "ted_afdb50_cath_dimerdb"
    )
    extract_and_assemble_dimers(
        manifest_path,
        db_path,
        output / "dimers",
        chunk_size=chunk_size,
        workers=workers,
        foldseek_threads=foldseek_threads,
    )
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
        download_nvidia_chunks,
        download_nvidia_metadata,
        extract_nvidia_structures,
        filter_nvidia_metadata,
    )

    csv_path = download_nvidia_metadata(output / "metadata")
    manifest_path = output / "filtered_manifest.tsv"
    filter_nvidia_metadata(
        csv_path,
        manifest_path,
        min_ipsae=min_ipsae,
        min_plddt=min_plddt,
        max_clashes=max_clashes,
    )
    download_nvidia_chunks(manifest_path, output / "chunks")
    extract_nvidia_structures(manifest_path, output / "chunks", output / "structures")
    typer.echo(f"NVIDIA complexes data prepared in {output}")


@download_app.command("prepare-manifests")
def prepare_manifests_cmd(
    output: Annotated[Path, typer.Option(help="Output directory for manifests.")] = Path(
        "data/manifests"
    ),
    teddymer: Annotated[Path | None, typer.Option(help="Teddymer filtered manifest.")] = None,
    nvidia: Annotated[Path | None, typer.Option(help="NVIDIA filtered manifest.")] = None,
    pdb: Annotated[Path | None, typer.Option(help="PDB complexes manifest.")] = None,
    val_fraction: Annotated[float, typer.Option(help="Validation fraction.")] = 0.05,
    seed: Annotated[int, typer.Option(help="Random seed for splitting.")] = 42,
) -> None:
    """Prepare unified train/val manifests with reproducible splits."""
    from teddympnn.data.splits import prepare_manifests

    train_path, val_path = prepare_manifests(
        output,
        teddymer_manifest=teddymer,
        nvidia_manifest=nvidia,
        pdb_manifest=pdb,
        val_fraction=val_fraction,
        seed=seed,
    )
    typer.echo(f"Train manifest: {train_path}")
    typer.echo(f"Val manifest:   {val_path}")
    typer.echo(f"Source manifests written to {output}")


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


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(help="Path to training config YAML."),
    ] = Path("configs/train.yaml"),
    resume: Annotated[
        Path | None,
        typer.Option(help="Checkpoint to resume from."),
    ] = None,
) -> None:
    """Train a teddyMPNN model.

    Extra positional arguments are treated as Hydra-style overrides
    (e.g. ``model.hidden_dim=256 data.train.teddymer.ratio=0.5``).
    """
    from teddympnn.config import load_training_config
    from teddympnn.training.trainer import Trainer

    training_config = load_training_config(config, ctx.args)
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
    checkpoint: Annotated[Path, typer.Option(help="teddyMPNN checkpoint path.")] = ...,  # type: ignore[assignment]
    output: Annotated[Path, typer.Option(help="Output Foundry checkpoint path.")] = ...,  # type: ignore[assignment]
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
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,  # type: ignore[assignment]
    data: Annotated[Path, typer.Option(help="Test data manifest path.")] = ...,  # type: ignore[assignment]
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
    interface_cutoff: Annotated[
        float, typer.Option(help="CB-CB distance cutoff for interface residues (A).")
    ] = 8.0,
) -> None:
    """Evaluate interface sequence recovery on a test set."""
    import torch

    from teddympnn.data.collator import PaddingCollator
    from teddympnn.data.dataset import PPIDataset
    from teddympnn.data.sampler import TokenBudgetBatchSampler
    from teddympnn.evaluation.sequence_recovery import compute_recovery
    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.io import load_model_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_model_weights(checkpoint, model, map_location=device)
    model = model.to(device)

    dataset = PPIDataset(data, include_ligand_atoms=(model_type == "ligand_mpnn"))
    collator = PaddingCollator()
    sampler = TokenBudgetBatchSampler(dataset.lengths, token_budget=10_000, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    results = compute_recovery(model, loader, interface_cutoff=interface_cutoff, device=device)

    typer.echo(f"Structures evaluated: {results.n_structures}")
    typer.echo(
        f"Overall recovery:     {results.overall_recovery:.4f} "
        f"({results.n_designed_residues} residues)"
    )
    typer.echo(
        f"Interface recovery:   {results.interface_recovery:.4f} "
        f"({results.n_interface_residues} residues)"
    )
    typer.echo(f"Per-structure (macro): {results.per_structure_recovery:.4f}")
    typer.echo(f"Per-struct interface:  {results.per_structure_interface_recovery:.4f}")
    if results.size_bin_recoveries:
        typer.echo("Interface size bins:")
        for name, rec in sorted(results.size_bin_recoveries.items()):
            typer.echo(f"  {name}: {rec:.4f}")


@evaluate_app.command()
def benchmark(
    config: Annotated[Path, typer.Option(help="Benchmark config YAML.")] = ...,  # type: ignore[assignment]
    output: Annotated[Path | None, typer.Option(help="Output JSON report path.")] = None,
) -> None:
    """Run benchmarks across multiple models and print comparison tables.

    The config YAML should have the structure::

        models:
          - name: "vanilla ProteinMPNN"
            checkpoint: "weights/v_48_020.pt"
            model_type: "protein_mpnn"
          - name: "teddyMPNN run1"
            checkpoint: "outputs/run1/checkpoints/step_0300000.pt"
            model_type: "protein_mpnn"

        test_manifests:
          teddymer: "data/manifests/val_manifest.tsv"
          pdb: "data/manifests/val_manifest.tsv"

        skempi_dir: "data/skempi"
        num_samples: 20
    """
    import yaml

    from teddympnn.evaluation.benchmark import (
        ModelSpec,
        print_comparison_table,
        run_benchmark,
    )

    with open(config) as f:
        cfg = yaml.safe_load(f)

    models = [ModelSpec(**m) for m in cfg["models"]]

    test_manifests = None
    if "test_manifests" in cfg:
        test_manifests = {k: Path(v) for k, v in cfg["test_manifests"].items()}

    skempi_dir = Path(cfg["skempi_dir"]) if cfg.get("skempi_dir") else None
    num_samples = cfg.get("num_samples", 20)

    report = run_benchmark(
        models,
        test_manifests=test_manifests,
        skempi_dir=skempi_dir,
        num_samples=num_samples,
    )

    print_comparison_table(report)

    if output is not None:
        report.save_json(output)
        typer.echo(f"Report saved to {output}")


@evaluate_app.command()
def ddg(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,  # type: ignore[assignment]
    skempi: Annotated[Path, typer.Option(help="SKEMPI data directory.")] = ...,  # type: ignore[assignment]
    num_samples: Annotated[int, typer.Option(help="Monte Carlo samples.")] = 20,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
    noise: Annotated[float, typer.Option(help="Backbone noise for scoring (A).")] = 0.0,
    max_entries: Annotated[int | None, typer.Option(help="Limit entries (for testing).")] = None,
) -> None:
    """Evaluate binding affinity prediction on SKEMPI v2.0."""
    import torch

    from teddympnn.evaluation.skempi import evaluate_skempi
    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.io import load_model_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_model_weights(checkpoint, model, map_location=device)
    model = model.to(device)

    results = evaluate_skempi(
        model,
        skempi,
        num_samples=num_samples,
        structure_noise=noise,
        device=device,
        max_entries=max_entries,
    )

    typer.echo(
        f"SKEMPI v2.0 Results ({results.n_entries} entries, {results.n_structures} structures)"
    )
    typer.echo(f"  Spearman:  {results.spearman:.4f}")
    typer.echo(f"  Pearson:   {results.pearson:.4f}")
    typer.echo(f"  RMSE:      {results.rmse:.4f}")
    typer.echo(f"  MAE:       {results.mae:.4f}")
    typer.echo(f"  AUROC:     {results.auroc:.4f}")
    typer.echo(f"  Per-struct Spearman (median): {results.per_structure_spearman_median:.4f}")


# ---------------------------------------------------------------------------
# Score command
# ---------------------------------------------------------------------------


@app.command()
def score(
    checkpoint: Annotated[Path, typer.Option(help="Model checkpoint path.")] = ...,  # type: ignore[assignment]
    pdb: Annotated[Path, typer.Option(help="PDB/mmCIF structure file.")] = ...,  # type: ignore[assignment]
    chains: Annotated[str, typer.Option(help="Design chain IDs (comma-separated).")] = ...,  # type: ignore[assignment]
    num_samples: Annotated[int, typer.Option(help="Monte Carlo samples.")] = 1,
    model_type: Annotated[
        str, typer.Option(help="Model type: protein_mpnn or ligand_mpnn.")
    ] = "protein_mpnn",
) -> None:
    """Score a structure with a trained model."""
    import torch

    from teddympnn.data.features import (
        derive_backbone,
        extract_ligand_atoms,
        extract_sidechain_atoms,
        parse_structure,
    )
    from teddympnn.models import LigandMPNN, ProteinMPNN
    from teddympnn.weights.io import load_model_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_model_weights(checkpoint, model, map_location=device)
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
    if model_type == "ligand_mpnn":
        ligand = extract_ligand_atoms(pdb)
        sidechains = extract_sidechain_atoms(
            features["xyz_37"],
            features["xyz_37_m"],
            features["S"],
            ~designed,
        )
        Y = ligand["Y"]
        Y_m = ligand["Y_m"]
        Y_t = ligand["Y_t"]
        if sidechains["Y"].shape[0] > 0:
            Y = torch.cat([Y, sidechains["Y"]], dim=0)
            Y_m = torch.cat([Y_m, sidechains["Y_m"]], dim=0)
            Y_t = torch.cat([Y_t, sidechains["Y_t"]], dim=0)
        batch["Y"] = Y.unsqueeze(0).to(device)
        batch["Y_m"] = Y_m.unsqueeze(0).to(device)
        batch["Y_t"] = Y_t.unsqueeze(0).to(device)

    scores = []
    for _ in range(num_samples):
        per_residue = model.score(batch, score_mask=designed.unsqueeze(0).to(device))
        total = per_residue.sum().item()
        scores.append(total)

    import statistics

    mean_score = statistics.mean(scores)
    typer.echo(f"Log-likelihood score: {mean_score:.4f} (n={num_samples})")
