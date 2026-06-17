"""
retrieval/conformal/calibrator.py — Split conformal calibration.

The calibrator learns a **nonconformity threshold** from a held-out
calibration set so that future predictions achieve a user-specified
**coverage guarantee** (e.g. 90 %).

Algorithm (standard split conformal prediction):
    1. Compute nonconformity scores for every example in the
       calibration set.
    2. Sort the scores ascending.
    3. Pick the score at the ``ceil((n+1) * (1 - α))``-th quantile
       as the threshold, where ``α = 1 - coverage_level`` and
       ``n`` is the calibration-set size.
    4. At inference time, include all packets whose nonconformity
       score ≤ threshold.

Coverage Guarantee (Angelopoulos & Bates, 2022, Theorem 1)
----------------------------------------------------------
For any exchangeable set of calibration nonconformity scores
:math:`V_1, \\dots, V_n` and any miscoverage level
:math:`\\alpha \\in (0, 1)`, the prediction set

.. math::

    C(X_{\\text{new}}) = \\{ y : V_{\\text{new}} \\leq
    \\hat{q} \\},
    \\qquad
    \\hat{q} = V_{(\\lceil (n+1)(1-\\alpha) \\rceil)}

satisfies

.. math::

    \\mathbb{P}\\big(Y_{\\text{new}} \\in C(X_{\\text{new}})\\big)
    \\geq 1 - \\alpha

where :math:`V_{(i)}` denotes the i-th order statistic of the
calibration scores.  This is **marginal coverage** — it holds on
average over the joint distribution of the calibration and test
data, requiring only that the calibration and test points are
exchangeable.  See Appendix A of Angelopoulos & Bates (2022) for
the proof using quantile concentration inequalities.

Reference:
    Angelopoulos, A. N., & Bates, S. (2022).
    "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification."
    arXiv:2107.07511.
"""

from __future__ import annotations

import math
from collections import defaultdict


class ConformalCalibrator:
    """Calibrates a nonconformity threshold for split conformal prediction.

    Implements split conformal prediction (Angelopoulos & Bates, 2022,
    Theorem 1).  The calibrated threshold :math:`\\hat{q}` is the
    :math:`\\lceil (n+1)(1-\\alpha) \\rceil`-th order statistic of
    the calibration nonconformity scores, where :math:`\\alpha = 1 -
    \\text{coverage_level}`.  Retaining all packets with
    nonconformity score :math:`\\leq \\hat{q}` guarantees:

    .. math::

        \\mathbb{P}(Y_{\\text{new}} \\in C(X_{\\text{new}}))
        \\geq 1 - \\alpha

    (marginal coverage over exchangeable calibration and test data).

    Args:
        coverage_level: Target coverage probability in (0, 1).
                        E.g. 0.90 means the prediction set should
                        contain the correct answer at least 90 % of
                        the time on held-out queries.
        min_calibration_size: Minimum number of calibration examples
            required.  If fewer are provided, calibration falls back
            to a conservative threshold of 0.0 (empty prediction set).

    Usage::

        calibrator = ConformalCalibrator(coverage_level=0.90)
        threshold = calibrator.calibrate(calibration_scores)
        # threshold can then be used with ConformalPredictor.predict()
    """

    def __init__(
        self,
        coverage_level: float = 0.90,
        min_calibration_size: int = 10,
    ) -> None:
        if not 0.0 < coverage_level < 1.0:
            raise ValueError(
                f"coverage_level must be in (0, 1), got {coverage_level}"
            )
        if min_calibration_size < 1:
            raise ValueError(
                f"min_calibration_size must be >= 1, got {min_calibration_size}"
            )
        self.coverage_level = coverage_level
        self.min_calibration_size = min_calibration_size
        self._last_threshold: float = 0.0

    def calibrate(self, nonconformity_scores: list[float]) -> float:
        """Compute the nonconformity threshold from a calibration set.

        Args:
            nonconformity_scores: Nonconformity scores from the
                calibration set (lower = more conforming).

        Returns:
            A threshold ``t`` such that retaining all packets with
            score ≤ ``t`` achieves the target coverage level.
            If the calibration set is too small, returns 0.0
            (conservative — no packet is retained).
        """
        n = len(nonconformity_scores)
        if n < self.min_calibration_size:
            # Too few examples — fall back conservatively
            return 0.0

        alpha = 1.0 - self.coverage_level
        # Rank index (1-based): ceil((n + 1) * (1 - α))
        # For standard split conformal prediction
        q_index = int(math.ceil((n + 1) * (1.0 - alpha)))

        # Clamp to valid range
        if q_index > n:
            q_index = n
        if q_index < 1:
            q_index = 1

        sorted_scores = sorted(nonconformity_scores)
        result = sorted_scores[q_index - 1]  # 0-based index
        self._last_threshold = result
        return result

    def estimate_coverage(
        self, scores: list[float], threshold: float
    ) -> float:
        """What fraction of the given scores fall ≤ threshold?

        This is useful for post-hoc evaluation on a test set.

        Args:
            scores:    Nonconformity scores to evaluate.
            threshold: The threshold from ``calibrate()``.

        Returns:
            Empirical coverage fraction in [0, 1].
        """
        if not scores:
            return 0.0
        covered = sum(1 for s in scores if s <= threshold)
        return covered / len(scores)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(coverage_level={self.coverage_level}, "
            f"min_calibration_size={self.min_calibration_size})"
        )


# ═══════════════════════════════════════════════════════════════════════
# MondorianConformalCalibrator
# ═══════════════════════════════════════════════════════════════════════


VALID_DOMAINS = frozenset({"LEGAL", "FINANCIAL", "TECHNICAL", "MEDICAL", "GENERAL"})


class MondorianConformalCalibrator:
    """Domain-stratified conformal calibrator with per-domain coverage guarantees.

    Extends split conformal prediction by maintaining a **separate threshold
    q_hat** for each domain (LEGAL, FINANCIAL, TECHNICAL, MEDICAL, GENERAL).
    This gives **conditional coverage** per domain rather than only marginal
    coverage across all queries.

    For each domain d, the coverage guarantee becomes::

        P(Y_new ∈ C_d(X_new) | domain = d) ≥ 1 - α

    where C_d uses the threshold :math:`\\hat{q}_d` calibrated on the d-th
    domain's hold-out scores.  This requires at least ``min_calibration_size``
    examples *per domain*.

    Usage::

        calibrator = MondorianConformalCalibrator(coverage_level=0.90)

        # Calibrate with labelled scores
        calibrator.add_domain_scores("LEGAL", [0.1, 0.2, 0.3, ...])
        calibrator.add_domain_scores("FINANCIAL", [0.15, 0.25, ...])

        # Compute per-domain thresholds
        thresholds = calibrator.calibrate_all()
        # => {"LEGAL": 0.35, "FINANCIAL": 0.28, ...}

        # Predict for a new query in the LEGAL domain
        retained = calibrator.predict("LEGAL", packets_with_scores)
    """

    def __init__(
        self,
        coverage_level: float = 0.90,
        min_calibration_size: int = 10,
    ) -> None:
        if not 0.0 < coverage_level < 1.0:
            raise ValueError(
                f"coverage_level must be in (0, 1), got {coverage_level}"
            )
        if min_calibration_size < 1:
            raise ValueError(
                f"min_calibration_size must be >= 1, got {min_calibration_size}"
            )
        self.coverage_level = coverage_level
        self.min_calibration_size = min_calibration_size
        self._domain_scores: dict[str, list[float]] = defaultdict(list)
        self._domain_thresholds: dict[str, float] = {}
        self._fitted: bool = False

    # ── Data collection ───────────────────────────────────────────────

    def add_domain_scores(
        self,
        domain: str,
        nonconformity_scores: list[float],
    ) -> None:
        """Add calibration scores for a specific domain.

        Args:
            domain:              One of LEGAL, FINANCIAL, TECHNICAL,
                                 MEDICAL, GENERAL.
            nonconformity_scores: Nonconformity scores from calibration
                                  queries in this domain.

        Raises:
            ValueError: If the domain is not recognised.
        """
        domain_upper = domain.upper()
        if domain_upper not in VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. Valid domains: {sorted(VALID_DOMAINS)}"
            )
        self._domain_scores[domain_upper].extend(nonconformity_scores)
        self._fitted = False

    # ── Calibration ───────────────────────────────────────────────────

    def calibrate_all(self) -> dict[str, float]:
        """Compute per-domain thresholds from all collected scores.

        For each domain with enough calibration examples, the standard
        split conformal q_hat is computed.  Domains with insufficient
        examples get a threshold of 0.0 (conservative — nothing retained).

        Returns:
            A dict mapping domain → threshold.
        """
        alpha = 1.0 - self.coverage_level
        thresholds: dict[str, float] = {}

        for domain in sorted(VALID_DOMAINS):
            scores = self._domain_scores.get(domain, [])
            n = len(scores)

            if n < self.min_calibration_size:
                thresholds[domain] = 0.0
                continue

            q_index = int(math.ceil((n + 1) * (1.0 - alpha)))
            if q_index > n:
                q_index = n
            if q_index < 1:
                q_index = 1

            sorted_scores = sorted(scores)
            thresholds[domain] = sorted_scores[q_index - 1]

        self._domain_thresholds = thresholds
        self._fitted = True
        return dict(thresholds)

    def get_threshold(self, domain: str) -> float:
        """Get the calibrated threshold for a single domain.

        Args:
            domain: One of LEGAL, FINANCIAL, TECHNICAL, MEDICAL, GENERAL.

        Returns:
            The threshold q_hat for this domain, or 0.0 if not calibrated.
        """
        domain_upper = domain.upper()
        if not self._fitted:
            self.calibrate_all()
        return self._domain_thresholds.get(domain_upper, 0.0)

    # ── Prediction ────────────────────────────────────────────────────

    def predict(
        self,
        domain: str,
        packets_with_scores: list[tuple[float, object]],
    ) -> list[tuple[float, object]]:
        """Filter packets for a given domain using its calibrated threshold.

        Args:
            domain:               One of LEGAL, FINANCIAL, etc.
            packets_with_scores:  List of ``(nonconformity_score, packet)`` tuples.

        Returns:
            The subset of tuples whose score ≤ the domain's threshold.
        """
        if not self._fitted:
            self.calibrate_all()

        threshold = self._domain_thresholds.get(domain.upper(), 0.0)
        return [(s, p) for s, p in packets_with_scores if s <= threshold]

    # ── Domain coverage estimation ────────────────────────────────────

    def estimate_coverage(
        self,
        domain_scores: dict[str, list[float]],
    ) -> dict[str, float]:
        """Estimate empirical coverage per domain on held-out scores.

        Args:
            domain_scores:  Dict mapping domain → list of nonconformity
                            scores from a held-out test set.

        Returns:
            Dict mapping domain → empirical coverage fraction in [0, 1].
        """
        if not self._fitted:
            self.calibrate_all()

        coverage: dict[str, float] = {}
        for domain, scores in domain_scores.items():
            domain_upper = domain.upper()
            threshold = self._domain_thresholds.get(domain_upper, 0.0)
            if not scores:
                coverage[domain_upper] = 0.0
            else:
                covered = sum(1 for s in scores if s <= threshold)
                coverage[domain_upper] = covered / len(scores)
        return coverage

    def __repr__(self) -> str:
        n_domains = len(self._domain_scores)
        return (
            f"{type(self).__name__}"
            f"(coverage_level={self.coverage_level}, "
            f"domains_loaded={n_domains}, "
            f"fitted={self._fitted})"
        )
