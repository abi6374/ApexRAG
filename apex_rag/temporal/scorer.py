"""
temporal/scorer.py — Exponential-decay freshness scoring.

Computes a ``freshness_score`` for each node using the formula::

    freshness = exp(-decay_rate * days_since_source_date)

where ``days_since_source_date`` is the number of days between the
node's ``source_date`` and a reference date (typically today).

Default decay rates by domain:

    - **General** documents: 0.001 per day (approx. 2.7 years to 50 % decay)
    - **News / market data**: 0.005 per day (approx. 5 months to 50 % decay)
    - **Legal / regulatory**: 0.0002 per day (approx. 9.5 years to 50 % decay)
"""

from __future__ import annotations

import math
from datetime import datetime, date, timezone

# ═══════════════════════════════════════════════════════════════
# Domain registry
# ═══════════════════════════════════════════════════════════════

DEFAULT_DECAY_RATES: dict[str, float] = {
    "general": 0.001,
    "news": 0.005,
    "market": 0.005,
    "financial": 0.005,
    "legal": 0.0002,
    "regulatory": 0.0002,
    "technical": 0.001,
    "scientific": 0.001,
    "medical": 0.0005,
    "educational": 0.001,
}

# ═══════════════════════════════════════════════════════════════
# FreshnessScorer
# ═══════════════════════════════════════════════════════════════


class FreshnessScorer:
    """Computes an exponential-decay freshness score for documents or nodes.

    The freshness score is defined as::

        freshness = exp(-decay_rate * days_since_source_date)

    where ``days_since_source_date`` is the number of calendar days between
    the source date and the reference date.  If the source date is in the
    future, the freshness is capped at 1.0.

    Usage::

        scorer = FreshnessScorer(domain="legal")
        score = scorer.compute(source_date=datetime(2024, 6, 1))
        # => exp(-0.0002 * days_since_2024-06-01)
    """

    def __init__(
        self,
        domain: str = "general",
        decay_rate: float | None = None,
        reference_date: date | None = None,
    ) -> None:
        """
        Args:
            domain:       A domain key from :data:`DEFAULT_DECAY_RATES`.
                          Ignored if ``decay_rate`` is explicitly provided.
            decay_rate:   Override the domain-based decay rate.
            reference_date: The date to decay "toward" (default: today).
        """
        self._decay_rate = (
            decay_rate if decay_rate is not None else DEFAULT_DECAY_RATES.get(domain, 0.001)
        )
        self._reference_date = reference_date or datetime.now(timezone.utc).date()

    @property
    def decay_rate(self) -> float:
        """The configured decay rate."""
        return self._decay_rate

    # ── Public API ─────────────────────────────────────────────────────────

    def compute(self, source_date: datetime | date | None) -> float:
        """Compute the freshness score for a given source date.

        Args:
            source_date: The date when the document was authored or published.
                         If ``None``, returns 1.0 (assumed current).

        Returns:
            A float in [0, 1] where 1 = just published, 0 = completely decayed.
        """
        if source_date is None:
            return 1.0

        if isinstance(source_date, datetime):
            source_date_d = source_date.date()
        else:
            source_date_d = source_date

        days = (self._reference_date - source_date_d).days

        if days <= 0:
            return 1.0

        freshness = math.exp(-self._decay_rate * days)
        return max(0.0, min(1.0, freshness))

    def compute_many(self, source_dates: list[datetime | date | None]) -> list[float]:
        """Compute freshness scores for multiple dates in bulk.

        Args:
            source_dates: A list of source dates.

        Returns:
            A list of freshness scores in the same order.
        """
        return [self.compute(d) for d in source_dates]

    def half_life_days(self) -> float:
        """Return the number of days until freshness = 0.5 at the current rate.

        Half-life = ln(2) / decay_rate
        """
        if self._decay_rate <= 0:
            return float("inf")
        return math.log(2) / self._decay_rate

    @staticmethod
    def suggest_decay_rate(desired_half_life_days: float) -> float:
        """Suggest a decay rate for a desired half-life in days.

        Example::

            >>> FreshnessScorer.suggest_decay_rate(365)
            0.001898  # ~0.19 % per day for 1-year half-life
        """
        if desired_half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        return math.log(2) / desired_half_life_days
