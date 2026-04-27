"""Multi-model benchmarking and comparison.

Runs interface sequence recovery and SKEMPI ddG evaluation across multiple
checkpoints (fine-tuned + baselines) and produces comparison tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from teddympnn.evaluation.sequence_recovery import RecoveryResults, compute_recovery
from teddympnn.evaluation.skempi import SKEMPIResults, evaluate_skempi
from teddympnn.models import LigandMPNN, ProteinMPNN
from teddympnn.weights.io import load_model_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Specification for a model to benchmark.

    Attributes:
        name: Human-readable model name for display.
        checkpoint: Path to model checkpoint.
        model_type: ``"protein_mpnn"`` or ``"ligand_mpnn"``.
    """

    name: str
    checkpoint: Path
    model_type: str = "protein_mpnn"


@dataclass
class BenchmarkResult:
    """Results for a single model across all benchmarks.

    Attributes:
        model_name: Name of the model.
        recovery: Sequence recovery results (None if not run).
        skempi: SKEMPI ddG results (None if not run).
    """

    model_name: str
    recovery: RecoveryResults | None = None
    skempi: SKEMPIResults | None = None


@dataclass
class BenchmarkReport:
    """Full benchmark report across multiple models.

    Attributes:
        results: Per-model benchmark results.
        test_sets: Names of test sets evaluated for recovery.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    test_sets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        out: dict[str, Any] = {"test_sets": self.test_sets, "models": []}
        for r in self.results:
            entry: dict[str, Any] = {"model_name": r.model_name}
            if r.recovery is not None:
                entry["recovery"] = asdict(r.recovery)
            if r.skempi is not None:
                d = asdict(r.skempi)
                d.pop("per_structure_spearman", None)
                entry["skempi"] = d
            out["models"].append(entry)
        return out

    def save_json(self, path: str | Path) -> None:
        """Save benchmark report to JSON.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Saved benchmark report to %s", path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(
    spec: ModelSpec,
    device: torch.device,
) -> ProteinMPNN | LigandMPNN:
    """Load a model from a checkpoint specification.

    Args:
        spec: Model specification with checkpoint path and type.
        device: Device to load model onto.

    Returns:
        Loaded model in eval mode.
    """
    model_cls = LigandMPNN if spec.model_type == "ligand_mpnn" else ProteinMPNN
    model = model_cls()
    load_model_weights(spec.checkpoint, model, map_location=device)
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    models: list[ModelSpec],
    *,
    test_manifests: dict[str, Path] | None = None,
    skempi_dir: Path | None = None,
    num_samples: int = 20,
    interface_cutoff: float = 8.0,
    token_budget: int = 10_000,
    device: torch.device | None = None,
) -> BenchmarkReport:
    """Run benchmarks across multiple models.

    Evaluates each model on interface sequence recovery (if test manifests
    are provided) and SKEMPI ddG prediction (if skempi_dir is provided).

    Args:
        models: List of model specifications to benchmark.
        test_manifests: ``{name: path}`` mapping of test set names to manifest
            paths for sequence recovery evaluation.
        skempi_dir: Directory containing (or to download) SKEMPI v2.0 data.
        num_samples: Monte Carlo samples for ddG prediction.
        interface_cutoff: CB-CB distance cutoff for interface residues (A).
        token_budget: Maximum residues per evaluation batch.
        device: Device for computation. Defaults to CUDA if available.

    Returns:
        BenchmarkReport with results for all models and test sets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = BenchmarkReport()
    if test_manifests:
        report.test_sets = list(test_manifests.keys())

    for spec in models:
        logger.info("Benchmarking model: %s (%s)", spec.name, spec.checkpoint)
        model = _load_model(spec, device)
        result = BenchmarkResult(model_name=spec.name)

        # Sequence recovery on test manifests
        if test_manifests:
            # Use first test manifest for the summary recovery result
            # (could be extended to per-test-set results)
            from teddympnn.data.collator import PaddingCollator
            from teddympnn.data.dataset import PPIDataset
            from teddympnn.data.sampler import TokenBudgetBatchSampler

            for name, manifest_path in test_manifests.items():
                logger.info("  Recovery on %s: %s", name, manifest_path)
                is_ligand = spec.model_type == "ligand_mpnn"
                dataset = PPIDataset(manifest_path, include_ligand_atoms=is_ligand)
                collator = PaddingCollator()
                sampler = TokenBudgetBatchSampler(
                    dataset.lengths, token_budget=token_budget, shuffle=False
                )
                loader = torch.utils.data.DataLoader(
                    dataset, batch_sampler=sampler, collate_fn=collator
                )
                recovery = compute_recovery(
                    model, loader, interface_cutoff=interface_cutoff, device=device
                )
                # Store last (or only) result as the summary
                result.recovery = recovery
                logger.info(
                    "    Overall=%.4f, Interface=%.4f (%d structures)",
                    recovery.overall_recovery,
                    recovery.interface_recovery,
                    recovery.n_structures,
                )

        # SKEMPI ddG
        if skempi_dir is not None:
            logger.info("  SKEMPI evaluation: %s", skempi_dir)
            skempi_results = evaluate_skempi(
                model,
                skempi_dir,
                num_samples=num_samples,
                device=device,
            )
            result.skempi = skempi_results
            logger.info(
                "    Spearman=%.4f, Pearson=%.4f, RMSE=%.4f",
                skempi_results.spearman,
                skempi_results.pearson,
                skempi_results.rmse,
            )

        report.results.append(result)

        # Free GPU memory between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return report


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_comparison_table(report: BenchmarkReport) -> None:
    """Print a formatted comparison table to the console.

    Args:
        report: Benchmark report with results for multiple models.
    """
    console = Console()

    has_recovery = any(r.recovery is not None for r in report.results)
    has_skempi = any(r.skempi is not None for r in report.results)

    if has_recovery:
        table = Table(title="Interface Sequence Recovery")
        table.add_column("Model", style="bold")
        table.add_column("Overall", justify="right")
        table.add_column("Interface", justify="right")
        table.add_column("Per-struct", justify="right")
        table.add_column("Per-struct IF", justify="right")
        table.add_column("Structures", justify="right")

        for r in report.results:
            if r.recovery is None:
                continue
            rec = r.recovery
            table.add_row(
                r.model_name,
                f"{rec.overall_recovery:.4f}",
                f"{rec.interface_recovery:.4f}",
                f"{rec.per_structure_recovery:.4f}",
                f"{rec.per_structure_interface_recovery:.4f}",
                str(rec.n_structures),
            )

        console.print(table)
        console.print()

    if has_skempi:
        table = Table(title="SKEMPI v2.0 Binding Affinity")
        table.add_column("Model", style="bold")
        table.add_column("Spearman", justify="right")
        table.add_column("Pearson", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("AUROC", justify="right")
        table.add_column("Per-struct Sp.", justify="right")

        for r in report.results:
            if r.skempi is None:
                continue
            sk = r.skempi
            table.add_row(
                r.model_name,
                f"{sk.spearman:.4f}",
                f"{sk.pearson:.4f}",
                f"{sk.rmse:.4f}",
                f"{sk.mae:.4f}",
                f"{sk.auroc:.4f}",
                f"{sk.per_structure_spearman_median:.4f}",
            )

        console.print(table)
