"""
retrieval/conformal/ — Conformal Prediction Layer (Part 5).

Provides uncertainty quantification for retrieved evidence via
split conformal prediction:

    - :class:`NonconformityScorer`     — Assigns nonconformity scores to
      evidence packets (lower = more conforming / stronger evidence).
    - :class:`ConformalCalibrator`     — Learns a threshold from a held-out
      calibration set at a target coverage level (e.g. 90 %).
    - :class:`MondorianConformalCalibrator` — Domain-stratified variant that
      maintains per-domain thresholds for conditional coverage.
    - :class:`ConformalPredictor`      — High-level API that wraps scoring
      and calibration into a single ``predict`` call.
    - :class:`CoverageVerifier`        — Empirically verifies the coverage
      guarantee on held-out test sets.
"""

from apex_rag.retrieval.conformal.calibrator import (
    ConformalCalibrator,
    MondorianConformalCalibrator,
)
from apex_rag.retrieval.conformal.coverage import CoverageVerifier
from apex_rag.retrieval.conformal.predictor import ConformalPredictor
from apex_rag.retrieval.conformal.scorer import NonconformityScorer, NonconformityStrategy

__all__ = [
    "NonconformityScorer",
    "NonconformityStrategy",
    "ConformalCalibrator",
    "MondorianConformalCalibrator",
    "ConformalPredictor",
    "CoverageVerifier",
]
