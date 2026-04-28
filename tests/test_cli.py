"""Tests for the teddyMPNN CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
    def test_train_default_config_path(self) -> None:
        """When --config is omitted, the loader is called with configs/train.yaml."""
        with (
            patch("teddympnn.config.load_training_config") as mock_load,
            patch("teddympnn.training.trainer.Trainer.from_config") as mock_from_config,
        ):
            mock_load.return_value = object()
            mock_from_config.return_value.train.return_value = None
            result = runner.invoke(app, ["train"])

        assert result.exit_code == 0, result.output
        assert mock_load.called
        passed_config_path, passed_overrides = mock_load.call_args.args
        assert passed_config_path == Path("configs/train.yaml")
        assert passed_overrides == []

    def test_train_passes_overrides(self) -> None:
        """Extra CLI args are forwarded to the loader as Hydra-style overrides."""
        with (
            patch("teddympnn.config.load_training_config") as mock_load,
            patch("teddympnn.training.trainer.Trainer.from_config") as mock_from_config,
        ):
            mock_load.return_value = object()
            mock_from_config.return_value.train.return_value = None
            result = runner.invoke(
                app,
                ["train", "model.hidden_dim=256", "data.train.teddymer.ratio=0.5"],
            )

        assert result.exit_code == 0, result.output
        _passed_config_path, passed_overrides = mock_load.call_args.args
        assert passed_overrides == ["model.hidden_dim=256", "data.train.teddymer.ratio=0.5"]

    def test_train_explicit_config(self, tmp_path: Path) -> None:
        """An explicit --config path is forwarded to the loader."""
        explicit = tmp_path / "explicit.yaml"
        with (
            patch("teddympnn.config.load_training_config") as mock_load,
            patch("teddympnn.training.trainer.Trainer.from_config") as mock_from_config,
        ):
            mock_load.return_value = object()
            mock_from_config.return_value.train.return_value = None
            result = runner.invoke(app, ["train", "--config", str(explicit)])

        assert result.exit_code == 0, result.output
        passed_config_path, _ = mock_load.call_args.args
        assert passed_config_path == explicit
