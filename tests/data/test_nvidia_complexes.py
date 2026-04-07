"""Tests for NVIDIA complexes data acquisition pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used in fixture type annotations

import pandas as pd
import pytest

from teddympnn.data.nvidia_complexes import filter_nvidia_metadata


class TestFilterNvidiaMetadata:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample metadata CSV for testing."""
        csv_path = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "model_id": [f"model_{i}" for i in range(10)],
                "ipSAEmin": [0.3, 0.5, 0.7, 0.8, 0.9, 0.4, 0.6, 0.65, 0.75, 0.1],
                "pLDDTavg": [80, 65, 75, 90, 85, 50, 72, 80, 60, 70],
                "N_clash_backbone": [2, 5, 8, 1, 0, 15, 3, 10, 7, 20],
                "chunk_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "filename": [f"model_{i}.cif" for i in range(10)],
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_filters_by_ipsae(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_nvidia_metadata(sample_csv, output)
        assert all(df["ipSAEmin"] >= 0.6)

    def test_filters_by_plddt(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_nvidia_metadata(sample_csv, output)
        assert all(df["pLDDTavg"] >= 70)

    def test_filters_by_clashes(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_nvidia_metadata(sample_csv, output)
        assert all(df["N_clash_backbone"] <= 10)

    def test_combined_filters(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_nvidia_metadata(sample_csv, output)

        # Expected passing entries:
        # model_2: ipSAE=0.7, pLDDT=75, clashes=8 ✓
        # model_3: ipSAE=0.8, pLDDT=90, clashes=1 ✓
        # model_4: ipSAE=0.9, pLDDT=85, clashes=0 ✓
        # model_6: ipSAE=0.6, pLDDT=72, clashes=3 ✓
        # model_7: ipSAE=0.65, pLDDT=80, clashes=10 ✓
        # model_8: ipSAE=0.75, pLDDT=60<70 ✗
        assert len(df) == 5

    def test_writes_output(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        filter_nvidia_metadata(sample_csv, output)
        assert output.exists()

    def test_custom_thresholds(self, sample_csv: Path, tmp_path: Path):
        output = tmp_path / "filtered.tsv"
        df = filter_nvidia_metadata(
            sample_csv,
            output,
            min_ipsae=0.8,
            min_plddt=80,
            max_clashes=5,
        )
        # model_3: ipSAE=0.8, pLDDT=90, clashes=1 ✓
        # model_4: ipSAE=0.9, pLDDT=85, clashes=0 ✓
        assert len(df) == 2
