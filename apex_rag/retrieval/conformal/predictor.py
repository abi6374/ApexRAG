"""
retrieval/conformal/predictor.py — High-level conformal prediction API.

Wraps :class:`NonconformityScorer` and :class:`ConformalCalibrator`
into a single ``predict`` call that returns a filtered prediction set
with a guaranteed coverage level.
"""

from __future__ import annotations

from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket
from apex_rag.models.unified_models import (
    EvidencePacket as UnifiedEvidencePacket,
)
from apex_rag.retrieval.conformal.calibrator import ConformalCalibrator
from apex_rag.retrieval.conformal.scorer import (
    NonconformityScorer,
    NonconformityStrategy,
)


class ConformalPredictor:
    """High-level conformal prediction for evidence packet filtering.

    Combines a :class:`NonconformityScorer` and a :class:`ConformalCalibrator`
    to produce prediction sets with statistical coverage guarantees.

    Args:
        scorer:      Nonconformity scorer instance.
        calibrator:  Conformal calibrator instance.

    Usage::

        predictor = ConformalPredictor()
        # Phase 1 — calibrate from a held-out set
        threshold = predictor.calibrate(calibration_scores)
        # Phase 2 — predict on new evidence
        filtered, guarantee, set_size = predictor.predict(packets, threshold)
    """

    def __init__(
        self,
        scorer: NonconformityScorer | None = None,
        calibrator: ConformalCalibrator | None = None,
    ) -> None:
        self.scorer = scorer or NonconformityScorer(strategy=NonconformityStrategy.ENSEMBLE)
        self.calibrator = calibrator or ConformalCalibrator(coverage_level=0.90)

    # ── Phase 1: Calibration ──────────────────────────────────────────

    def calibrate(
        self,
        calibration_scores: list[float],
    ) -> float:
        """Calibrate the threshold using scores from a held-out set.

        Args:
            calibration_scores: Nonconformity scores from the
                calibration set.

        Returns:
            The threshold value.
        """
        return self.calibrator.calibrate(calibration_scores)

    # ── Phase 2: Prediction ───────────────────────────────────────────

    def predict(
        self,
        packets: list[CoreEvidencePacket] | list[UnifiedEvidencePacket],
        threshold: float,
    ) -> tuple[
        list[CoreEvidencePacket | UnifiedEvidencePacket],
        float,
        int,
    ]:
        """Filter packets by the calibrated threshold.

        Args:
            packets:   The evidence packets to filter.
            threshold: The nonconformity threshold from ``calibrate()``.

        Returns:
            A tuple of ``(filtered_packets, coverage_guarantee, prediction_set_size)``.
            - ``filtered_packets``: Packets whose nonconformity score ≤ threshold.
            - ``coverage_guarantee``: The target coverage level (from calibrator).
            - ``prediction_set_size``: Number of retained packets.
        """
        if not packets:
            return [], self.calibrator.coverage_level, 0

        # Compute nonconformity scores
        # (CoreEvidencePacket vs UnifiedEvidencePacket both have
        #  confidence_score, verification_result, etc.)
        scores = self.scorer.score_many(packets)  # type: ignore[arg-type]

        # Filter by threshold
        filtered: list[CoreEvidencePacket | UnifiedEvidencePacket] = [
            pkt
            for pkt, s in zip(packets, scores, strict=False)
            if s <= threshold  # type: ignore[misc]
        ]

        return (
            filtered,
            self.calibrator.coverage_level,
            len(filtered),
        )

    # ── End-to-end: calibrate + predict ───────────────────────────────

    def fit_predict(
        self,
        calibration_scores: list[float],
        packets: list[CoreEvidencePacket] | list[UnifiedEvidencePacket],
    ) -> tuple[
        list[CoreEvidencePacket | UnifiedEvidencePacket],
        float,
        int,
    ]:
        """Convenience: calibrate then predict in one call.

        Args:
            calibration_scores: Scores from the calibration set.
            packets:            Evidence packets to filter.

        Returns:
            Same as ``predict()``.
        """
        threshold = self.calibrate(calibration_scores)
        return self.predict(packets, threshold)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scorer={self.scorer!r}, calibrator={self.calibrator!r})"
