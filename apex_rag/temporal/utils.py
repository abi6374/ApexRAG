"""
temporal/utils.py — Shared utility functions for the temporal module.

Provides common helpers used across temporal sub-modules to avoid code
duplication.  All functions are deterministic and stateless.

Functions:
    windows_overlap:  Check if two temporal windows overlap.
"""

from __future__ import annotations

from datetime import datetime


def windows_overlap(
    from_a: datetime | None,
    to_a: datetime | None,
    from_b: datetime | None,
    to_b: datetime | None,
) -> bool:
    """Check if two temporal windows overlap.

    A window [valid_from, valid_to] overlaps with another if:
      - a.valid_from <= b.valid_to (or b.valid_to is None)
      - AND (a.valid_to is None OR a.valid_to > b.valid_from)

    Args:
        from_a: Start of first window (or None).
        to_a:   End of first window (or None for open-ended).
        from_b: Start of second window (or None).
        to_b:   End of second window (or None for open-ended).

    Returns:
        True if the two windows overlap in time.
    """
    if from_a is None or from_b is None:
        return False

    a_end = to_a or datetime.max.replace(tzinfo=from_a.tzinfo)
    b_end = to_b or datetime.max.replace(tzinfo=from_b.tzinfo)

    return from_a <= b_end and a_end > from_b
