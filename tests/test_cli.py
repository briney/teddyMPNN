"""Tests for the teddyMPNN CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from teddympnn.cli import app

runner = CliRunner()


class TestCLIHelp:
    def test_main_help(self) -> None:
        """Main --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "teddympnn" in result.output.lower() or "protein" in result.output.lower()

    def test_version(self) -> None:
        """--version shows version string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "teddympnn" in result.output

    def test_download_help(self) -> None:
        """download --help works."""
        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0

    def test_download_teddymer_help(self) -> None:
        """download teddymer --help works."""
        result = runner.invoke(app, ["download", "teddymer", "--help"])
        assert result.exit_code == 0

    def test_download_nvidia_help(self) -> None:
        """download nvidia-complexes --help works."""
        result = runner.invoke(app, ["download", "nvidia-complexes", "--help"])
        assert result.exit_code == 0

    def test_download_prepare_manifests_help(self) -> None:
        """download prepare-manifests --help works."""
        result = runner.invoke(app, ["download", "prepare-manifests", "--help"])
        assert result.exit_code == 0
        assert "val-fraction" in result.output.lower() or "validation" in result.output.lower()

    def test_download_pretrained_help(self) -> None:
        """download pretrained --help works."""
        result = runner.invoke(app, ["download", "pretrained", "--help"])
        assert result.exit_code == 0

    def test_train_help(self) -> None:
        """train --help works."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_checkpoints_help(self) -> None:
        """checkpoints --help works."""
        result = runner.invoke(app, ["checkpoints", "--help"])
        assert result.exit_code == 0

    def test_checkpoints_export_foundry_help(self) -> None:
        """checkpoints export-foundry --help works."""
        result = runner.invoke(app, ["checkpoints", "export-foundry", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self) -> None:
        """evaluate --help works."""
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_evaluate_recovery_help(self) -> None:
        """evaluate recovery --help works."""
        result = runner.invoke(app, ["evaluate", "recovery", "--help"])
        assert result.exit_code == 0

    def test_evaluate_benchmark_help(self) -> None:
        """evaluate benchmark --help works."""
        result = runner.invoke(app, ["evaluate", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_evaluate_ddg_help(self) -> None:
        """evaluate ddg --help works."""
        result = runner.invoke(app, ["evaluate", "ddg", "--help"])
        assert result.exit_code == 0

    def test_score_help(self) -> None:
        """score --help works."""
        result = runner.invoke(app, ["score", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.output.lower()


class TestTrainCommand:
    def test_train_missing_config(self) -> None:
        """train without --config flag fails."""
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0
