"""observability/_stats.py — Shared statistics helpers for the observability layer.

Provides reusable percentile and histogram summary functions used by
both :class:`MetricsService` and :class:`AccuracyTracker` to avoid
duplicated logic.
"""

from __future__ import annotations


def percentile_summary(
    values: list[float],
    *,
    round_digits: int = 4,
) -> dict[str, float]:
    """Compute percentile summary from a list of values.

    Args:
        values:       List of float values to summarise.
        round_digits: Number of decimal places for rounding (default 4).

    Returns:
        A dict with keys: mean, min, max, p50, p95, p99.
        All zero when *values* is empty.
    """
    if not values:
        return zero_percentile_summary()

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "mean": round(sum(sorted_vals) / n, round_digits),
        "min": round(sorted_vals[0], round_digits),
        "max": round(sorted_vals[-1], round_digits),
        "p50": round(sorted_vals[int(n * 0.50)], round_digits),
        "p95": round(sorted_vals[int(n * 0.95)], round_digits),
        "p99": round(sorted_vals[int(n * 0.99)], round_digits),
    }


def histogram_summary(values: list[float], *, round_digits: int = 2) -> dict[str, float]:
    """Compute histogram statistics from a list of values.

    Extends :func:`percentile_summary` with count and sum fields,
    suitable for Prometheus-style histogram reporting.

    Args:
        values:       List of float values to summarise.
        round_digits: Number of decimal places for rounding (default 2).

    Returns:
        A dict with keys: count, sum, avg, min, max, p50, p95, p99.
        All zero when *values* is empty.
    """
    if not values:
        return _zero_histogram()

    p = percentile_summary(values, round_digits=round_digits)
    n = len(values)
    return {
        "count": n,
        "sum": round(sum(values), round_digits),
        "avg": p["mean"],
        "min": p["min"],
        "max": p["max"],
        "p50": p["p50"],
        "p95": p["p95"],
        "p99": p["p99"],
    }


def zero_percentile_summary() -> dict[str, float]:
    """Return a zero-filled percentile summary dict."""
    return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def _zero_histogram() -> dict[str, float]:
    """Return a zero-filled histogram summary dict."""
    return {
        "count": 0.0,
        "sum": 0.0,
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
    }
