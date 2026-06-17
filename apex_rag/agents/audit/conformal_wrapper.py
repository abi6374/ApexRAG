"""
agents/audit/conformal_wrapper.py — ConformalWrapperAgent.

Wraps the ConformalCalibrator and NonconformityScorer into a standalone
agent interface that the Orchestrator can call to produce the coverage
guarantee and prediction set size fields of the ApexAnswer.

Usage::

    wrapper = ConformalWrapperAgent(coverage_level=0.90)
    result = await wrapper.wrap(packets, calibration_scores)
    # => (filtered_packets, coverage_guarantee, prediction_set_size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from apex_rag.models.unified_models import EvidencePacket
from apex_rag.retrieval.conformal.calibrator import (
    ConformalCalibrator,
    MondorianConformalCalibrator,
)
from apex_rag.retrieval.conformal.scorer import (
    NonconformityScorer,
    NonconformityStrategy,
)

logger = logging.getLogger("apex_rag.agents.audit.conformal")


@dataclass
class ConformalResult:
    """Result of applying conformal prediction to a set of evidence packets.

    Attributes:
        filtered_packets:    Packets retained by the conformal threshold.
        coverage_guarantee:  The target coverage level (e.g. 0.90).
        prediction_set_size: Number of retained packets.
        nonconformity_scores: List of (packet, score) for all input packets.
    """

    filtered_packets: list[EvidencePacket]
    coverage_guarantee: float
    prediction_set_size: int
    nonconformity_scores: list[tuple[EvidencePacket, float]]


class ConformalWrapperAgent:
    """Wraps conformal prediction infrastructure as a callable agent.

    This agent:
    1. Computes nonconformity scores for all evidence packets.
    2. Calibrates a threshold from a held-out calibration set (if provided).
    3. Filters packets whose nonconformity score ≤ threshold.
    4. Returns a :class:`ConformalResult` with the filtered set and guarantee.

    Can be used standalone, or chained through the Orchestrator for the
    final ConformalWrapper step of the pipeline.
    """

    def __init__(
        self,
        calibrator: ConformalCalibrator | None = None,
        scorer: NonconformityScorer | None = None,
        coverage_level: float = 0.90,
    ) -> None:
        self._calibrator = calibrator or ConformalCalibrator(
            coverage_level=coverage_level,
        )
        self._scorer = scorer or NonconformityScorer(
            strategy=NonconformityStrategy.ENSEMBLE,
        )
        self._last_threshold: float = 0.0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def calibrator(self) -> ConformalCalibrator | MondorianConformalCalibrator:
        """The underlying calibrator instance."""
        return self._calibrator

    @property
    def scorer(self) -> NonconformityScorer:
        """The underlying nonconformity scorer."""
        return self._scorer

    # ── Public API ────────────────────────────────────────────────────

    def calibrate(self, calibration_scores: list[float]) -> float:
        """Calibrate the conformal threshold from a held-out set.

        Args:
            calibration_scores: Nonconformity scores from the calibration set.

        Returns:
            The calibrated threshold value.
        """
        self._last_threshold = self._calibrator.calibrate(calibration_scores)
        return self._last_threshold

    def wrap(
        self,
        packets: list[EvidencePacket],
        calibration_scores: list[float] | None = None,
    ) -> ConformalResult:
        """Filter evidence packets through the conformal prediction set.

        Args:
            packets:            The evidence packets to filter.
            calibration_scores: Optional — if provided, calibrate first.
                                If omitted, uses the last calibrated threshold
                                (or a conservative 0.0 if never calibrated).

        Returns:
            A :class:`ConformalResult` with filtered packets and guarantee.
        """
        if not packets:
            return ConformalResult(
                filtered_packets=[],
                coverage_guarantee=self._calibrator.coverage_level,
                prediction_set_size=0,
                nonconformity_scores=[],
            )

        # 1. Calibrate if scores provided
        if calibration_scores is not None:
            self.calibrate(calibration_scores)

        threshold = self._last_threshold

        # 2. Compute nonconformity scores
        scores = self._scorer.score_many(packets)  # type: ignore[arg-type]

        # 3. Apply threshold
        scored_packets: list[tuple[EvidencePacket, float]] = [
            (pkt, s) for pkt, s in zip(packets, scores, strict=False)
        ]
        filtered = [
            pkt for pkt, s in scored_packets if s <= threshold
        ]

        # If threshold is 0.0 (no calibration), keep all packets but
        # still report the nonconformity scores as metadata
        if threshold == 0.0 and not filtered:
            filtered = list(packets)
            logger.info(
                "ConformalWrapper: threshold=0.0 (uncalibrated), "
                "keeping all %d packets",
                len(packets),
            )

        return ConformalResult(
            filtered_packets=filtered,
            coverage_guarantee=self._calibrator.coverage_level
            if threshold > 0.0
            else 0.0,
            prediction_set_size=len(filtered),
            nonconformity_scores=scored_packets,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(coverage_level={self._calibrator.coverage_level}, "
            f"last_threshold={self._last_threshold:.4f})"
        )
