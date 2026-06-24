"""
retrieval/conformal/scorer.py — Nonconformity scoring strategies.

Nonconformity scores quantify *how unusual* an evidence packet is
relative to the expected pattern.  Lower scores = more conforming
(= stronger evidence).  Scores are always non-negative.

Available strategies (see :class:`NonconformityStrategy`):

    - ``INVERSE_RETRIEVAL``  : 1.0 - retrieval_score
    - ``VERIFICATION_GAP``   : 1.0 - verifier_confidence
    - ``RANK_BASED``         : rank / max_rank
    - ``ENSEMBLE``           : weighted combination of the above
"""

from __future__ import annotations

from enum import Enum

from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket


class NonconformityStrategy(str, Enum):
    """Available nonconformity scoring strategies."""

    INVERSE_RETRIEVAL = "inverse_retrieval"
    VERIFICATION_GAP = "verification_gap"
    RANK_BASED = "rank_based"
    ENSEMBLE = "ensemble"


# Weight defaults for the ensemble strategy
_DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    NonconformityStrategy.INVERSE_RETRIEVAL.value: 0.40,
    NonconformityStrategy.VERIFICATION_GAP.value: 0.35,
    NonconformityStrategy.RANK_BASED.value: 0.25,
}


class NonconformityScorer:
    """Computes nonconformity scores for evidence packets.

    Args:
        strategy:     Scoring strategy to use.
        weights:      Per-strategy weights for ``ENSEMBLE``
                      (ignored for other strategies).
        max_rank:     Maximum possible rank for ``RANK_BASED``
                      normalisation.  If ``None``, inferred from the
                      number of packets at call time.

    Usage::

        scorer = NonconformityScorer(strategy="inverse_retrieval")
        scores = scorer.score_many(packets)
    """

    def __init__(
        self,
        strategy: str | NonconformityStrategy = NonconformityStrategy.INVERSE_RETRIEVAL,
        weights: dict[str, float] | None = None,
        max_rank: int | None = None,
    ) -> None:
        if isinstance(strategy, str):
            strategy = NonconformityStrategy(strategy)
        self.strategy = strategy
        self.weights = weights or dict(_DEFAULT_ENSEMBLE_WEIGHTS)
        self._max_rank = max_rank

    # ── Public API ────────────────────────────────────────────────────

    def score(self, packet: CoreEvidencePacket, max_rank: int = 1) -> float:
        """Compute the nonconformity score for a single packet.

        Args:
            packet:   The evidence packet to score.
            max_rank: Maximum rank among the full packet set
                      (used only by ``RANK_BASED``).

        Returns:
            A non-negative float; lower is more conforming.
        """
        if self.strategy == NonconformityStrategy.INVERSE_RETRIEVAL:
            return self._inverse_retrieval(packet)
        if self.strategy == NonconformityStrategy.VERIFICATION_GAP:
            return self._verification_gap(packet)
        if self.strategy == NonconformityStrategy.RANK_BASED:
            return self._rank_based(packet, max_rank)
        if self.strategy == NonconformityStrategy.ENSEMBLE:
            return self._ensemble(packet, max_rank)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def score_many(self, packets: list[CoreEvidencePacket]) -> list[float]:
        """Compute nonconformity scores for a batch of packets.

        Args:
            packets: The evidence packets to score.

        Returns:
            A list of non-negative floats, one per packet.
            Lower = more conforming.
        """
        max_rank = self._max_rank or len(packets)
        return [self.score(p, max_rank=max_rank) for p in packets]

    # ── Per-strategy implementations ───────────────────────────────────

    @staticmethod
    def _get_score(packet: CoreEvidencePacket) -> float:
        """Extract the retrieval/confidence score from a packet.

        CoreEvidencePacket uses ``confidence_score``, while
        UnifiedEvidencePacket uses ``retrieval_score``.  This helper
        checks both so the scorer works with either type.
        """
        score = getattr(packet, "confidence_score", None)
        if score is not None:
            return score  # type: ignore[no-any-return]
        return getattr(packet, "retrieval_score", 0.0)

    @staticmethod
    def _get_rank(packet: CoreEvidencePacket) -> int:
        """Extract the rank from a packet.

        Returns ``packet.rank`` (UnifiedEvidencePacket has this field)
        or 1 as a default for CoreEvidencePacket which lacks it.
        """
        return getattr(packet, "rank", 1)

    @staticmethod
    def _inverse_retrieval(packet: CoreEvidencePacket) -> float:
        """1.0 - retrieval_score, clamped to [0, ∞)."""
        score = NonconformityScorer._get_score(packet)
        return max(0.0, 1.0 - score)

    @staticmethod
    def _verification_gap(packet: CoreEvidencePacket) -> float:
        """1.0 - verification_result (as float), clamped to [0, ∞).

        If verification_result is False, the gap is 1.0 (maximally
        nonconforming).  If True, the gap is 0.0 (perfectly conforming).

        Supports both :class:`CoreEvidencePacket` (has ``verification_result``)
        and :class:`UnifiedEvidencePacket` (no ``verification_result`` —
        assumed verified since only Critic-approved packets reach the scorer).
        """
        verified = getattr(packet, "verification_result", None)
        if verified is None:
            # UnifiedEvidencePacket — assume verified
            return 0.0
        return 0.0 if verified else 1.0

    @staticmethod
    def _rank_based(packet: CoreEvidencePacket, max_rank: int) -> float:
        """rank / max_rank — normalised position in the evidence set."""
        if max_rank <= 0:
            return 1.0
        rank = NonconformityScorer._get_rank(packet)
        return min(1.0, rank / max_rank)

    def _ensemble(
        self,
        packet: CoreEvidencePacket,
        max_rank: int,
    ) -> float:
        """Weighted sum of all three base strategies."""
        raw = (
            self._inverse_retrieval(packet)
            * self.weights.get(NonconformityStrategy.INVERSE_RETRIEVAL.value, 0.4)
            + self._verification_gap(packet)
            * self.weights.get(NonconformityStrategy.VERIFICATION_GAP.value, 0.35)
            + self._rank_based(packet, max_rank)
            * self.weights.get(NonconformityStrategy.RANK_BASED.value, 0.25)
        )
        return max(0.0, raw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(strategy={self.strategy.value})"
