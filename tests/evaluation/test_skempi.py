"""Tests for SKEMPI v2.0 benchmark utilities."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime in fixtures

import numpy as np
import pytest

from teddympnn.evaluation.skempi import (
    _parse_mutation_string,
    _parse_partner_chains,
    auroc,
    parse_skempi,
    pearson_correlation,
    spearman_correlation,
)

# ---------------------------------------------------------------------------
# Synthetic SKEMPI CSV for testing
# ---------------------------------------------------------------------------

_HEADER = (
    "#Pdb;Mutation(s)_PDB;iMutation_Location(s);Hold_out_type;"
    "Hold_out_proteins;Affinity_mut (M);Affinity_mut_parsed;"
    "Affinity_wt (M);Affinity_wt_parsed;Temperature;pH;ddG;"
    "Publication;Method;SKEMPI version"
)

_ROWS = [
    "1BRS_A_D;DA83G;COR;Pr;1BRS;5e-08;5e-08;1e-10;1e-10;298.15;8.0;3.27;S;SPR;2.0",
    "1BRS_A_D;DA39E;COR;Pr;1BRS;2e-09;2e-09;1e-10;1e-10;298.15;8.0;1.82;S;SPR;2.0",
    "1BRS_A_D;DA83G,DA39E;COR;Pr;1BRS;1e-07;1e-07;1e-10;1e-10;298.15;8.0;4.09;S;SPR;2.0",
    "3HFM_Y_AB;YA101G;INT;Pr;3HFM;5e-06;5e-06;1e-09;1e-09;298.15;7.4;5.05;L;ITC;2.0",
    "3HFM_Y_AB;YT30A;RIM;Pr;3HFM;1e-07;1e-07;1e-09;1e-09;298.15;7.4;2.73;L;ITC;2.0",
    "1BRS_A_D;DA83A;COR;Pr;1BRS;;bad;;bad;298.15;8.0;;S;SPR;2.0",
]

_SYNTHETIC_SKEMPI_CSV = _HEADER + "\n" + "\n".join(_ROWS) + "\n"


class TestParseSkempi:
    """Tests for SKEMPI CSV parsing."""

    def test_parses_valid_entries(self, tmp_path: Path) -> None:
        """Should parse entries with valid ddG and mutations."""
        csv_path = tmp_path / "skempi.csv"
        csv_path.write_text(_SYNTHETIC_SKEMPI_CSV)

        entries = parse_skempi(csv_path)

        # 5 valid entries (last row has missing ddG)
        assert len(entries) == 5

    def test_entry_fields(self, tmp_path: Path) -> None:
        """Parsed entries should have correct field values."""
        csv_path = tmp_path / "skempi.csv"
        csv_path.write_text(_SYNTHETIC_SKEMPI_CSV)

        entries = parse_skempi(csv_path)
        e = entries[0]

        assert e.pdb_id == "1BRS"
        assert e.partner_chains == ({"A"}, {"D"})
        assert e.ddg_experimental == pytest.approx(3.27)
        assert "D" in e.mutations
        assert "A83G" in e.mutations["D"]
        assert e.mutation_type == "single"

    def test_multi_mutation_entry(self, tmp_path: Path) -> None:
        """Multi-mutation entries should be tagged correctly."""
        csv_path = tmp_path / "skempi.csv"
        csv_path.write_text(_SYNTHETIC_SKEMPI_CSV)

        entries = parse_skempi(csv_path)
        multi = [e for e in entries if e.mutation_type == "multi"]
        assert len(multi) == 1
        # Should have 2 mutations on chain D
        assert len(multi[0].mutations["D"]) == 2

    def test_multi_chain_partner(self, tmp_path: Path) -> None:
        """Partners with multiple chains (e.g., AB) should be parsed correctly."""
        csv_path = tmp_path / "skempi.csv"
        csv_path.write_text(_SYNTHETIC_SKEMPI_CSV)

        entries = parse_skempi(csv_path)
        hfm_entries = [e for e in entries if e.pdb_id == "3HFM"]
        assert len(hfm_entries) == 2
        # 3HFM_Y_AB → partner 1 = {Y}, partner 2 = {A, B}
        assert hfm_entries[0].partner_chains == ({"Y"}, {"A", "B"})

    def test_skips_missing_ddg(self, tmp_path: Path) -> None:
        """Entries with missing or unparseable ddG should be skipped."""
        csv_path = tmp_path / "skempi.csv"
        csv_path.write_text(_SYNTHETIC_SKEMPI_CSV)

        entries = parse_skempi(csv_path)
        # Last row has empty ddG → should be skipped
        assert all(e.ddg_experimental != 0 for e in entries)


class TestParseMutationString:
    """Tests for individual mutation string parsing."""

    def test_standard_mutation(self) -> None:
        chain, mut = _parse_mutation_string("IA45G")
        assert chain == "I"
        assert mut == "A45G"

    def test_insertion_code(self) -> None:
        chain, mut = _parse_mutation_string("AL52aG")
        assert chain == "A"
        assert mut == "L52aG"

    def test_negative_residue_number(self) -> None:
        chain, mut = _parse_mutation_string("AC-3G")
        assert chain == "A"
        assert mut == "C-3G"

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_mutation_string("invalid")


class TestParsePartnerChains:
    """Tests for partner chain parsing."""

    def test_two_single_chains(self) -> None:
        pdb_id, (p1, p2) = _parse_partner_chains("1BRS_A_D")
        assert pdb_id == "1BRS"
        assert p1 == {"A"}
        assert p2 == {"D"}

    def test_multi_chain_partner(self) -> None:
        pdb_id, (p1, p2) = _parse_partner_chains("3HFM_Y_AB")
        assert pdb_id == "3HFM"
        assert p1 == {"Y"}
        assert p2 == {"A", "B"}


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestCorrelationMetrics:
    """Tests for numpy-based correlation and AUROC implementations."""

    def test_pearson_perfect_positive(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert pearson_correlation(x, y) == pytest.approx(1.0)

    def test_pearson_perfect_negative(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        assert pearson_correlation(x, y) == pytest.approx(-1.0)

    def test_spearman_monotone(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 100.0, 200.0])  # monotone, not linear
        assert spearman_correlation(x, y) == pytest.approx(1.0)

    def test_spearman_with_ties(self) -> None:
        x = np.array([1.0, 2.0, 2.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        sp = spearman_correlation(x, y)
        assert 0.9 < sp <= 1.0

    def test_auroc_perfect_separation(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert auroc(y_true, y_scores) == pytest.approx(1.0)

    def test_auroc_chance(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=1000).astype(np.float64)
        y_scores = rng.randn(1000)
        auc = auroc(y_true, y_scores)
        assert 0.4 < auc < 0.6  # should be near 0.5

    def test_auroc_inverse(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert auroc(y_true, y_scores) == pytest.approx(0.0)


class TestSKEMPIResultsComputation:
    """Tests for metrics computation on synthetic predictions."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give Spearman/Pearson ≈ 1."""
        exp = np.array([1.0, 2.0, -0.5, 3.0, -1.0])
        pred = exp.copy()

        sp = spearman_correlation(exp, pred)
        pe = pearson_correlation(exp, pred)

        assert sp == pytest.approx(1.0)
        assert pe == pytest.approx(1.0)

    def test_rmse_mae_on_known_values(self) -> None:
        """RMSE and MAE on known differences."""
        exp = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 2.5, 3.5])

        rmse = float(np.sqrt(np.mean((exp - pred) ** 2)))
        mae = float(np.mean(np.abs(exp - pred)))

        assert rmse == pytest.approx(0.5)
        assert mae == pytest.approx(0.5)
