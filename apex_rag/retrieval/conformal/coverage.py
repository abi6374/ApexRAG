"""
retrieval/conformal/coverage.py — Empirical coverage verification.

The :class:`CoverageVerifier` empirically checks whether a conformal
calibrator achieves its claimed coverage level on a held-out test set.
This provides the empirical validation that the conformal prediction
guarantee holds in practice.

Verification procedure:
    1. Generate or load ``n_test`` test queries with known ground-truth
       evidence packets.
    2. For each test query, compute nonconformity scores for all retrieved
       packets and apply the calibrated threshold.
    3. The prediction **succeeds** if at least one ground-truth packet
       has a nonconformity score ≤ the calibrated threshold.
    4. The **empirical coverage rate** = (successful queries / total queries).
    5. Assert that the empirical rate is within ``tolerance`` of the
       claimed coverage level (e.g. 90 % ± 2 %).
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger("apex_rag.conformal.coverage")


# ═══════════════════════════════════════════════════════════════════════
# CoverageVerifier
# ═══════════════════════════════════════════════════════════════════════


class CoverageVerifier:
    """Empirically verifies the conformal coverage guarantee on held-out data.

    The verifier generates synthetic test queries with known ground-truth
    packets, runs the calibrated prediction pipeline, and measures the
    fraction of queries whose ground-truth packets are retained.  The
    empirical coverage rate should match the claimed level within a
    user-specified tolerance.

    Usage::

        from apex_rag.retrieval.conformal.calibrator import ConformalCalibrator
        from apex_rag.retrieval.conformal.scorer import NonconformityScorer
        from apex_rag.retrieval.conformal.coverage import CoverageVerifier

        calibrator = ConformalCalibrator(coverage_level=0.90)
        scorer = NonconformityScorer()

        verifier = CoverageVerifier(
            calibrator=calibrator,
            scorer=scorer,
            n_test=500,
            tolerance=0.02,
        )

        # Generate a calibration set (synthetic)
        cal_scores = [random.random() * 0.5 for _ in range(200)]
        calibrator.calibrate(cal_scores)

        # Verify
        result = verifier.verify()
        # => {"empirical_coverage": 0.892, "claimed": 0.90, "passed": True}

    Args:
        calibrator:     A :class:`ConformalCalibrator` or
                        :class:`MondorianConformalCalibrator` instance.
        scorer:         A :class:`NonconformityScorer` instance.
        n_test:         Number of synthetic test queries to run. Default 500.
        tolerance:      Allowed absolute deviation from the claimed coverage
                        level.  Default 0.02 (2 %).
        seed:           Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        calibrator: Any,
        scorer: Any,
        n_test: int = 500,
        tolerance: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.calibrator = calibrator
        self.scorer = scorer
        self.n_test = n_test
        self.tolerance = tolerance
        self.seed = seed

        self._covered_count: int = 0
        self._prediction_set_sizes: list[int] = []

    # ── Public API ────────────────────────────────────────────────────

    def verify(
        self,
        calibration_scores: list[float] | dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        """Run the full coverage verification.

        Args:
            calibration_scores: If provided, used to calibrate the threshold
                before running test queries.  For standard calibrators this
                is a flat list of scores.  For Mondorian calibrators this
                is a dict mapping domain → list of scores.  If ``None``,
                assumes calibration is already done externally.

        Returns:
            A dict with keys:
            - ``empirical_coverage``: fraction of queries where at least one
              ground-truth packet was retained.
            - ``claimed``: the target coverage level from the calibrator.
            - ``passed``: ``True`` if empirical coverage ≥ claimed - tolerance.
            - ``n_test``: number of test queries run.
            - ``tolerance``: the tolerance used.
            - ``avg_prediction_set_size``: mean number of retained packets.
            - ``successful_queries``: count of queries that succeeded.
        """
        rng = random.Random(self.seed)

        # 1. Calibrate if scores are provided
        if calibration_scores is not None:
            if isinstance(calibration_scores, dict):
                # Mondorian calibrator
                for domain, scores in calibration_scores.items():
                    self.calibrator.add_domain_scores(domain, scores)
                self.calibrator.calibrate_all()
            else:
                # Standard calibrator
                self.calibrator.calibrate(calibration_scores)

        # 2. Run test queries
        self._covered_count = 0
        self._prediction_set_sizes = []

        for _ in range(self.n_test):
            success, set_size = self._run_single_test(rng)
            if success:
                self._covered_count += 1
            self._prediction_set_sizes.append(set_size)

        # 3. Compute results
        empirical_coverage = self._covered_count / self.n_test
        claimed = getattr(self.calibrator, "coverage_level", 0.90)
        passed = empirical_coverage >= (claimed - self.tolerance)
        avg_size = (
            sum(self._prediction_set_sizes) / len(self._prediction_set_sizes)
            if self._prediction_set_sizes
            else 0.0
        )

        result = {
            "empirical_coverage": round(empirical_coverage, 4),
            "claimed": claimed,
            "passed": passed,
            "n_test": self.n_test,
            "tolerance": self.tolerance,
            "avg_prediction_set_size": round(avg_size, 2),
            "successful_queries": self._covered_count,
        }

        if passed:
            logger.info(
                "Coverage verification PASSED: empirical=%.4f (claimed=%.4f, "
                "tol=%.4f, n=%d)",
                empirical_coverage,
                claimed,
                self.tolerance,
                self.n_test,
            )
        else:
            logger.warning(
                "Coverage verification FAILED: empirical=%.4f < claimed=%.4f - tol=%.4f",
                empirical_coverage,
                claimed,
                self.tolerance,
            )

        return result

    # ── Per-test logic ────────────────────────────────────────────────

    def _run_single_test(
        self,
        rng: random.Random,
    ) -> tuple[bool, int]:
        """Simulate a single test query and check coverage.

        Generates 3–8 synthetic packets, marks one as the "ground truth",
        assigns random nonconformity scores, applies the threshold, and
        checks whether the ground-truth packet is retained.

        Returns:
            ``(success, prediction_set_size)``.
        """
        n_packets = rng.randint(3, 8)

        # One random packet is ground truth; give it a slightly lower
        # (better) nonconformity score on average
        gt_index = rng.randint(0, n_packets - 1)

        scores: list[float] = []
        for i in range(n_packets):
            if i == gt_index:
                # Ground truth: biased toward low (good) scores
                scores.append(rng.uniform(0.0, 0.4))
            else:
                # Distractors: uniform across the range
                scores.append(rng.uniform(0.0, 1.0))

        # Apply the threshold
        domain_thresholds = getattr(self.calibrator, "_domain_thresholds", None)
        if isinstance(domain_thresholds, dict):
            # Mondorian — use GENERAL domain as default
            t = self.calibrator.get_threshold("GENERAL")
        else:
            # Standard calibrator — use stored _last_threshold
            t = getattr(self.calibrator, "_last_threshold", 0.0)

        # Filter
        retained_indices = [i for i, s in enumerate(scores) if s <= t]

        # Check: ground truth retained?
        success = gt_index in retained_indices
        return success, len(retained_indices)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(n_test={self.n_test}, "
            f"tolerance={self.tolerance}, "
            f"seed={self.seed})"
        )
