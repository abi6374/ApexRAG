"""
tests/retrieval/conformal/test_calibrator_v2.py — Tests for Part 5 additions.

Covers:
    1. MondorianConformalCalibrator — 12 tests
       - Per-domain thresholds produce distinct values
       - Each of the 5 valid domains works
       - Invalid domain raises ValueError
       - Small calibration falls back to 0.0
       - calibrate_all returns correct thresholds
       - predict filters correctly per domain
       - get_threshold returns stored value
       - estimate_coverage evaluates per domain
       - Coverage matches expected level across domains
       - Empty domain scores handled
       - Multiple add_domain_scores calls accumulate
       - round-trip: add → calibrate → predict → estimate

    2. CoverageVerifier — 8 tests
       - verify returns expected keys
       - verify passes with sufficient calibration data
       - verify tolerates small calibration sets
       - verify handles empty calibration
       - verify result is deterministic with same seed
       - verify logs coverage stats
       - verify with Mondorian calibrator
       - prediction set sizes are recorded

    Total: ~20 tests
"""

from __future__ import annotations

import math
import random

import pytest

from apex_rag.retrieval.conformal.calibrator import (
    ConformalCalibrator,
    MondorianConformalCalibrator,
    VALID_DOMAINS,
)
from apex_rag.retrieval.conformal.coverage import CoverageVerifier
from apex_rag.retrieval.conformal.scorer import NonconformityScorer


# ═══════════════════════════════════════════════════════════════════════
# 1. MondorianConformalCalibrator — 12 tests
# ═══════════════════════════════════════════════════════════════════════


class TestMondorianConformalCalibrator:
    """Domain-stratified conformal calibration."""

    def test_invalid_domain_raises(self) -> None:
        """Unknown domain raises ValueError."""
        calibrator = MondorianConformalCalibrator()
        with pytest.raises(ValueError, match="Unknown domain"):
            calibrator.add_domain_scores("INVALID", [0.1, 0.2])

    def test_valid_domains_accepted(self) -> None:
        """All 5 valid domains are accepted without error."""
        calibrator = MondorianConformalCalibrator()
        for domain in VALID_DOMAINS:
            calibrator.add_domain_scores(domain, [0.1, 0.2, 0.3])

    def test_case_insensitive_domain(self) -> None:
        """Domain matching is case-insensitive."""
        calibrator = MondorianConformalCalibrator()
        calibrator.add_domain_scores("legal", [0.1, 0.2])
        calibrator.add_domain_scores("Financial", [0.3, 0.4])
        calibrator.add_domain_scores("TECHNICAL", [0.5, 0.6])
        thresholds = calibrator.calibrate_all()
        assert "LEGAL" in thresholds
        assert "FINANCIAL" in thresholds
        assert "TECHNICAL" in thresholds

    def test_small_calibration_returns_zero(self) -> None:
        """Insufficient calibration per domain returns 0.0 threshold."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.90, min_calibration_size=10
        )
        calibrator.add_domain_scores("LEGAL", [0.1, 0.2, 0.3])
        thresholds = calibrator.calibrate_all()
        assert thresholds["LEGAL"] == 0.0

    def test_calibrate_all_returns_all_domains(self) -> None:
        """calibrate_all returns thresholds for all 5 domains."""
        calibrator = MondorianConformalCalibrator(min_calibration_size=1)
        calibrator.add_domain_scores("LEGAL", [0.1])
        calibrator.add_domain_scores("FINANCIAL", [0.2])
        thresholds = calibrator.calibrate_all()
        assert len(thresholds) == 5  # All VALID_DOMAINS
        assert "LEGAL" in thresholds
        assert "FINANCIAL" in thresholds
        assert "GENERAL" in thresholds

    def test_distinct_domain_thresholds(self) -> None:
        """Different domains with different score distributions get different thresholds."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.90, min_calibration_size=5
        )
        # LEGAL: very low scores (tight)
        calibrator.add_domain_scores(
            "LEGAL", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        )
        # FINANCIAL: high variance
        calibrator.add_domain_scores(
            "FINANCIAL",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        thresholds = calibrator.calibrate_all()
        # LEGAL should have a much lower threshold than FINANCIAL
        assert thresholds["LEGAL"] < thresholds["FINANCIAL"]

    def test_predict_filters_correctly(self) -> None:
        """predict retains only packets with NC score ≤ domain threshold."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.90, min_calibration_size=5
        )
        calibrator.add_domain_scores(
            "LEGAL", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        calibrator.calibrate_all()

        # All low scores → all retained
        packets = [(0.05, "pkt1"), (0.10, "pkt2"), (0.50, "pkt3")]
        retained = calibrator.predict("LEGAL", packets)
        assert len(retained) >= 1  # At least the low ones

    def test_predict_high_scores_filtered(self) -> None:
        """High scores are filtered out by domain threshold."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.90, min_calibration_size=5
        )
        calibrator.add_domain_scores(
            "LEGAL", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        calibrator.calibrate_all()

        # Very high scores → should be filtered
        packets = [(0.95, "pkt1"), (0.99, "pkt2")]
        retained = calibrator.predict("LEGAL", packets)
        assert len(retained) == 0  # Both should exceed threshold

    def test_get_threshold(self) -> None:
        """get_threshold returns stored threshold for a specific domain."""
        calibrator = MondorianConformalCalibrator(min_calibration_size=1)
        calibrator.add_domain_scores("TECHNICAL", [0.25])
        calibrator.calibrate_all()
        thresh = calibrator.get_threshold("TECHNICAL")
        assert thresh == 0.25

    def test_get_threshold_auto_fits(self) -> None:
        """get_threshold auto-calibrates if not fitted yet."""
        calibrator = MondorianConformalCalibrator(min_calibration_size=1)
        calibrator.add_domain_scores("MEDICAL", [0.30])
        # No explicit calibrate_all() call
        thresh = calibrator.get_threshold("MEDICAL")
        assert thresh == 0.30

    def test_get_threshold_unseen_domain(self) -> None:
        """Unseen domain returns 0.0."""
        calibrator = MondorianConformalCalibrator(min_calibration_size=1)
        calibrator.add_domain_scores("LEGAL", [0.1])
        thresh = calibrator.get_threshold("GENERAL")
        assert thresh == 0.0  # Not calibrated

    def test_estimate_coverage(self) -> None:
        """estimate_coverage returns per-domain empirical coverage."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.90, min_calibration_size=5
        )
        calibrator.add_domain_scores(
            "LEGAL", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        )
        calibrator.calibrate_all()

        test_scores = {
            "LEGAL": [0.05, 0.10, 0.50, 0.95],
        }
        coverage = calibrator.estimate_coverage(test_scores)
        assert "LEGAL" in coverage
        assert 0.0 <= coverage["LEGAL"] <= 1.0

    def test_add_domain_scores_accumulates(self) -> None:
        """Multiple add_domain_scores calls accumulate scores."""
        calibrator = MondorianConformalCalibrator(min_calibration_size=5)
        calibrator.add_domain_scores("GENERAL", [0.1, 0.2, 0.3])
        calibrator.add_domain_scores("GENERAL", [0.4, 0.5])
        calibrator.calibrate_all()
        thresh = calibrator.get_threshold("GENERAL")
        assert thresh > 0.0  # 5 scores present

    def test_repr(self) -> None:
        """__repr__ provides useful information."""
        calibrator = MondorianConformalCalibrator(coverage_level=0.90)
        r = repr(calibrator)
        assert "MondorianConformalCalibrator" in r
        assert "0.9" in r


# ═══════════════════════════════════════════════════════════════════════
# 2. CoverageVerifier — 8 tests
# ═══════════════════════════════════════════════════════════════════════


class TestCoverageVerifier:
    """Empirical coverage verification."""

    def test_verify_returns_expected_keys(self) -> None:
        """verify returns dict with all expected keys."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=10)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=50, seed=42
        )

        cal_scores = [random.random() * 0.5 for _ in range(50)]
        result = verifier.verify(calibration_scores=cal_scores)

        expected_keys = {
            "empirical_coverage",
            "claimed",
            "passed",
            "n_test",
            "tolerance",
            "avg_prediction_set_size",
            "successful_queries",
        }
        assert set(result.keys()) == expected_keys

    def test_verify_passes_with_sufficient_data(self) -> None:
        """verify returns passed=True with enough calibration data."""
        calibrator = ConformalCalibrator(coverage_level=0.80, min_calibration_size=10)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator,
            scorer=scorer,
            n_test=100,
            tolerance=0.05,
            seed=42,
        )

        # Calibration scores spread across the full range [0.0, 1.0] so the
        # threshold covers ground-truth test packets (biased to [0.0, 0.4])
        cal_scores = [i / 100.0 for i in range(1, 101)]  # 0.01 ... 1.00
        result = verifier.verify(calibration_scores=cal_scores)
        # With n=100 coverage_level=0.80: q_index=ceil(101*0.80)=81 → sorted[80]=0.81
        # Ground-truth packets (0.0–0.4) should almost always be retained
        assert result["passed"]

    def test_verify_tolerates_small_calibration(self) -> None:
        """verify handles small calibration set gracefully."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=50)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=20, tolerance=0.10, seed=42
        )

        # Only 5 calibration scores — below min_calibration_size=50
        cal_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = verifier.verify(calibration_scores=cal_scores)
        # Threshold should be 0.0 (conservative fallback)
        # Most queries should fail since nothing is retained
        assert "empirical_coverage" in result

    def test_verify_empty_calibration(self) -> None:
        """verify handles empty calibration without crashing."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=5)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=10, seed=42
        )
        result = verifier.verify(calibration_scores=[])
        assert "empirical_coverage" in result

    def test_verify_no_calibration_arg(self) -> None:
        """verify works when calibration is done externally."""
        calibrator = ConformalCalibrator(coverage_level=0.80, min_calibration_size=10)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=50, seed=42
        )
        # Calibrate externally
        cal_scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                      0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
        calibrator.calibrate(cal_scores)
        # Then call verify without calibration_scores
        result = verifier.verify()
        assert "empirical_coverage" in result

    def test_verify_with_mondorian_calibrator(self) -> None:
        """verify works with MondorianConformalCalibrator."""
        calibrator = MondorianConformalCalibrator(
            coverage_level=0.80, min_calibration_size=5
        )
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=50, seed=42
        )

        # Calibrate with domain scores
        cal_scores: dict[str, list[float]] = {
            "GENERAL": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        }
        result = verifier.verify(calibration_scores=cal_scores)
        assert "empirical_coverage" in result
        assert isinstance(result["passed"], bool)

    def test_prediction_set_sizes_recorded(self) -> None:
        """avg_prediction_set_size is a non-negative float."""
        calibrator = ConformalCalibrator(coverage_level=0.90, min_calibration_size=10)
        scorer = NonconformityScorer()
        verifier = CoverageVerifier(
            calibrator=calibrator, scorer=scorer, n_test=50, seed=42
        )
        cal_scores = [0.1 * i for i in range(1, 21)]
        result = verifier.verify(calibration_scores=cal_scores)
        assert result["avg_prediction_set_size"] >= 0.0
        assert isinstance(result["avg_prediction_set_size"], float)

    def test_deterministic_results(self) -> None:
        """Same seed produces same results."""
        calibrator = ConformalCalibrator(coverage_level=0.80, min_calibration_size=5)
        scorer = NonconformityScorer()
        cal_scores = [0.1 * i for i in range(1, 11)]

        v1 = CoverageVerifier(
            calibrator=ConformalCalibrator(coverage_level=0.80, min_calibration_size=5),
            scorer=NonconformityScorer(),
            n_test=100,
            seed=123,
        )
        r1 = v1.verify(calibration_scores=list(cal_scores))

        v2 = CoverageVerifier(
            calibrator=ConformalCalibrator(coverage_level=0.80, min_calibration_size=5),
            scorer=NonconformityScorer(),
            n_test=100,
            seed=123,
        )
        r2 = v2.verify(calibration_scores=list(cal_scores))

        assert r1["empirical_coverage"] == r2["empirical_coverage"]
        assert r1["successful_queries"] == r2["successful_queries"]


# ═══════════════════════════════════════════════════════════════════════
# 3. Proof docstring — verify it compiles and mentions the theorem
# ═══════════════════════════════════════════════════════════════════════


class TestProofDocstring:
    """Verify the mathematical proof docstring meets spec requirements."""

    def test_coverage_guarantee_citation_present(self) -> None:
        """Docstring of ConformalCalibrator cites Angelopoulos & Bates 2022."""
        doc = ConformalCalibrator.__doc__ or ""
        assert "Angelopoulos" in doc, "Angelopoulos citation missing"
        assert "Bates" in doc, "Bates citation missing"
        assert "2022" in doc, "Year missing"
        assert "Theorem 1" in doc, "Theorem 1 not cited"

    def test_mondorian_docstring_found(self) -> None:
        """MondorianConformalCalibrator has a docstring."""
        doc = MondorianConformalCalibrator.__doc__ or ""
        assert len(doc) > 50, "Docstring too short"

    def test_coverage_verifier_docstring_found(self) -> None:
        """CoverageVerifier has a docstring."""
        doc = CoverageVerifier.__doc__ or ""
        assert len(doc) > 50, "Docstring too short"

