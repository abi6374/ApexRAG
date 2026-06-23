"""
core/metrics/parser.py — Metric Value Parser.

Parses human-readable metric values such as "$100k", "₹2Cr", "35%", "N/A"
into structured numeric representations with unit type and confidence.

Supports:
  - Currencies: $100, $100k, $2.5M, ₹10L, ₹2Cr
  - Percentages: 35%, -5%
  - Business units: million, billion, lakh, crore
  - Unknown values: N/A, Unknown, Not Available

Usage:
    parser = MetricValueParser()
    parsed = parser.parse("$100k")
    # ParsedMetric(numeric_value=100000.0, original_value="$100k",
    #              unit_type=CURRENCY, confidence=1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UnitType(str, Enum):
    """Enumeration of recognised unit types for parsed metrics."""

    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    NUMERIC = "NUMERIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedMetric:
    """Structured result of a metric value parse.

    Attributes:
        numeric_value:  The parsed numeric value in the canonical unit
                        (e.g. 100000 for $100k, 0.35 for 35%).
        original_value: The raw input string.
        unit_type:      The :class:`UnitType` classification.
        confidence:     Confidence score in [0, 1].
        metadata:       Optional additional parse metadata.
    """

    numeric_value: float
    original_value: str
    unit_type: UnitType
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Constants ─────────────────────────────────────────────────────────────

_CURRENCY_MULTIPLIERS: dict[str, float] = {
    "k": 1_000,
    "K": 1_000,
    "m": 1_000_000,
    "M": 1_000_000,
    "b": 1_000_000_000,
    "B": 1_000_000_000,
}

_INDIAN_MULTIPLIERS: dict[str, float] = {
    "l": 100_000,
    "L": 100_000,
    "cr": 100_000_00,
    "Cr": 100_000_00,
    "CR": 100_000_00,
}

_TEXT_MULTIPLIERS: dict[str, float] = {
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "thousand": 1_000,
    "lakh": 100_000,
    "crore": 10_000_000,
}

_UNKNOWN_VALUES: frozenset[str] = frozenset(
    {
        "n/a",
        "na",
        "unknown",
        "not available",
        "none",
        "null",
        "tbd",
        "t.b.d.",
        "not applicable",
    }
)

# Regex patterns
_RE_CURRENCY = re.compile(
    r"^\s*"                          # leading whitespace
    r"([₹$€£¥])?"                    # optional currency symbol (group 1)
    r"\s*"                           # optional whitespace
    r"(-?\d+(?:\.\d+)?)"            # numeric value (group 2)
    r"\s*"                           # optional whitespace
    r"([kKmMbBtTlLcCrR]|Cr|cr|CR)?"  # optional multiplier suffix (group 3)
    r"\s*$"                          # trailing whitespace
)

_RE_PERCENTAGE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$"
)

_RE_TEXT_MULTIPLIER = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s+("
    r"million|billion|thousand|lakh|crore"
    r")\s*$",
    re.IGNORECASE,
)


class MetricValueParser:
    """Parse human-readable metric values into structured :class:`ParsedMetric`.

    Thread-safe, stateless, and fully deterministic.
    """

    @staticmethod
    def parse(value: str) -> ParsedMetric:
        """Parse a single metric value string.

        Args:
            value: The raw string value to parse (e.g. "$100k", "35%", "N/A").

        Returns:
            A :class:`ParsedMetric` with the parsed result.  If the value
            cannot be recognised, returns ``ParsedMetric(0.0, value,
            UnitType.UNKNOWN, 0.0)``.
        """
        stripped = value.strip()

        if not stripped:
            return ParsedMetric(
                numeric_value=0.0,
                original_value=value,
                unit_type=UnitType.UNKNOWN,
                confidence=0.0,
            )

        # 1. Unknown / sentinel values
        if stripped.lower() in _UNKNOWN_VALUES:
            return ParsedMetric(
                numeric_value=0.0,
                original_value=value,
                unit_type=UnitType.UNKNOWN,
                confidence=0.0,
            )

        # 2. Percentage values
        m = _RE_PERCENTAGE.match(stripped)
        if m:
            num = float(m.group(1))
            return ParsedMetric(
                numeric_value=num / 100.0,
                original_value=value,
                unit_type=UnitType.PERCENTAGE,
                confidence=1.0,
            )

        # 3. Currency / numeric with multiplier suffixes
        m = _RE_CURRENCY.match(stripped)
        if m:
            symbol = m.group(1)
            num = float(m.group(2))
            suffix = m.group(3)

            multiplier = MetricValueParser._resolve_multiplier(
                suffix, symbol == "₹"
            )
            return ParsedMetric(
                numeric_value=num * multiplier,
                original_value=value,
                unit_type=UnitType.CURRENCY if symbol else UnitType.NUMERIC,
                confidence=1.0,
                metadata={
                    "currency_symbol": symbol or "",
                    "multiplier_suffix": suffix or "",
                    "multiplier_value": multiplier,
                },
            )

        # 4. Text-based multipliers ("5 million", "2 crore")
        m = _RE_TEXT_MULTIPLIER.match(stripped)
        if m:
            num = float(m.group(1))
            unit_word = m.group(2).lower()
            multiplier = _TEXT_MULTIPLIERS.get(unit_word, 1.0)
            return ParsedMetric(
                numeric_value=num * multiplier,
                original_value=value,
                unit_type=UnitType.NUMERIC,
                confidence=1.0,
                metadata={
                    "multiplier_word": unit_word,
                    "multiplier_value": multiplier,
                },
            )

        # 5. Plain numeric (try float conversion)
        try:
            return ParsedMetric(
                numeric_value=float(stripped),
                original_value=value,
                unit_type=UnitType.NUMERIC,
                confidence=0.9,
            )
        except ValueError:
            pass

        # 6. Fallback — unrecognised
        return ParsedMetric(
            numeric_value=0.0,
            original_value=value,
            unit_type=UnitType.UNKNOWN,
            confidence=0.0,
        )

    @staticmethod
    def _resolve_multiplier(suffix: str | None, is_indian: bool) -> float:
        """Resolve a suffix (k, M, B, L, Cr) to its numeric multiplier.

        Args:
            suffix:    The suffix string, or None.
            is_indian: True if the currency symbol is ₹ (Indian Rupee).

        Returns:
            The numeric multiplier.
        """
        if not suffix:
            return 1.0

        if is_indian:
            multiplier = _INDIAN_MULTIPLIERS.get(suffix)
            if multiplier is not None:
                return multiplier
            # Fall through to standard multipliers for non-Indian suffixes

        return _CURRENCY_MULTIPLIERS.get(suffix, 1.0)
