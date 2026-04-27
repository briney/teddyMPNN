"""Tests for the benchmark comparison tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from teddympnn.evaluation.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    ModelSpec,
    print_comparison_table,
)
from teddympnn.evaluation.sequence_recovery import RecoveryResults
from teddympnn.evaluation.skempi import SKEMPIResults


class TestBenchmarkResult:
    def test_empty_result(self) -> None:
        """BenchmarkResult can be created with no metrics."""
        result = BenchmarkResult(model_name="test")
        assert result.recovery == {}
        assert result.skempi is None

    def test_with_recovery(self) -> None:
        """BenchmarkResult stores recovery metrics keyed by test set."""
        recovery = RecoveryResults(
            overall_recovery=0.42,
            interface_recovery=0.38,
            per_structure_recovery=0.40,
            per_structure_interface_recovery=0.36,
            n_structures=100,
            n_designed_residues=5000,
            n_interface_residues=1200,
        )
        result = BenchmarkResult(model_name="test", recovery={"val": recovery})
        assert "val" in result.recovery
        assert result.recovery["val"].overall_recovery == pytest.approx(0.42)

    def test_multiple_test_sets(self) -> None:
        """BenchmarkResult preserves recovery for each test set."""
        rec_a = RecoveryResults(
            overall_recovery=0.40,
            interface_recovery=0.35,
            per_structure_recovery=0.38,
            per_structure_interface_recovery=0.33,
            n_structures=10,
            n_designed_residues=500,
            n_interface_residues=120,
        )
        rec_b = RecoveryResults(
            overall_recovery=0.55,
            interface_recovery=0.50,
            per_structure_recovery=0.52,
            per_structure_interface_recovery=0.48,
            n_structures=10,
            n_designed_residues=500,
            n_interface_residues=120,
        )
        result = BenchmarkResult(model_name="m", recovery={"val_a": rec_a, "val_b": rec_b})
        assert result.recovery["val_a"].overall_recovery == pytest.approx(0.40)
        assert result.recovery["val_b"].overall_recovery == pytest.approx(0.55)


class TestBenchmarkReport:
    @pytest.fixture()
    def sample_report(self) -> BenchmarkReport:
        """Create a sample report with two models."""
        results = [
            BenchmarkResult(
                model_name="baseline",
                recovery={
                    "val": RecoveryResults(
                        overall_recovery=0.40,
                        interface_recovery=0.35,
                        per_structure_recovery=0.38,
                        per_structure_interface_recovery=0.33,
                        n_structures=50,
                        n_designed_residues=3000,
                        n_interface_residues=800,
                    )
                },
                skempi=SKEMPIResults(
                    spearman=0.30,
                    pearson=0.28,
                    rmse=2.5,
                    mae=1.8,
                    auroc=0.65,
                    per_structure_spearman_median=0.25,
                    n_entries=100,
                    n_structures=20,
                ),
            ),
            BenchmarkResult(
                model_name="fine-tuned",
                recovery={
                    "val": RecoveryResults(
                        overall_recovery=0.50,
                        interface_recovery=0.48,
                        per_structure_recovery=0.49,
                        per_structure_interface_recovery=0.47,
                        n_structures=50,
                        n_designed_residues=3000,
                        n_interface_residues=800,
                    )
                },
                skempi=SKEMPIResults(
                    spearman=0.45,
                    pearson=0.42,
                    rmse=2.0,
                    mae=1.4,
                    auroc=0.72,
                    per_structure_spearman_median=0.40,
                    n_entries=100,
                    n_structures=20,
                ),
            ),
        ]
        return BenchmarkReport(results=results, test_sets=["val"])

    def test_to_dict(self, sample_report: BenchmarkReport) -> None:
        """Report serializes to dict with expected structure."""
        d = sample_report.to_dict()
        assert d["test_sets"] == ["val"]
        assert len(d["models"]) == 2
        assert d["models"][0]["model_name"] == "baseline"
        assert "recovery" in d["models"][0]
        assert "skempi" in d["models"][0]
        # per_structure_spearman should be stripped from SKEMPI
        assert "per_structure_spearman" not in d["models"][0]["skempi"]

    def test_save_json(self, sample_report: BenchmarkReport, tmp_path: Path) -> None:
        """Report saves to valid JSON."""
        json_path = tmp_path / "report.json"
        sample_report.save_json(json_path)
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert len(data["models"]) == 2
        assert data["models"][1]["recovery"]["val"]["overall_recovery"] == pytest.approx(0.50)

    def test_print_comparison_table(self, sample_report: BenchmarkReport) -> None:
        """Comparison table prints without errors."""
        # Just verify it doesn't raise
        print_comparison_table(sample_report)

    def test_empty_report(self) -> None:
        """Empty report serializes without errors."""
        report = BenchmarkReport()
        d = report.to_dict()
        assert d["models"] == []
        assert d["test_sets"] == []


class TestModelSpec:
    def test_defaults(self) -> None:
        """ModelSpec has sensible defaults."""
        spec = ModelSpec(name="test", checkpoint=Path("test.pt"))
        assert spec.model_type == "protein_mpnn"

    def test_ligand_mpnn(self) -> None:
        """ModelSpec accepts ligand_mpnn type."""
        spec = ModelSpec(name="test", checkpoint=Path("test.pt"), model_type="ligand_mpnn")
        assert spec.model_type == "ligand_mpnn"
