from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

logger = logging.getLogger("apex_rag.temporal.analyzers")

class ChangeAnalyzer:
    """
    ChangeAnalyzer performs comparisons between two distinct versions of content, metrics, or policies.
    """

    def compare_metrics(self, before_val: float, after_val: float) -> dict[str, Any]:
        """Compares two numeric metric values and computes absolute and percentage differences."""
        diff = after_val - before_val
        pct_change = (diff / before_val * 100.0) if before_val != 0.0 else 0.0
        direction = "increase" if diff > 0 else "decrease" if diff < 0 else "no_change"
        return {
            "before": before_val,
            "after": after_val,
            "difference": round(diff, 4),
            "percentage_change": round(pct_change, 2),
            "direction": direction
        }

    def compare_versions(self, before_text: str, after_text: str) -> dict[str, Any]:
        """Compares two text-based version changes to extract differences."""
        # Clean text comparison
        added = []
        removed = []
        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()

        before_set = set(before_lines)
        after_set = set(after_lines)

        for line in after_lines:
            if line not in before_set:
                added.append(line)
        for line in before_lines:
            if line not in after_set:
                removed.append(line)

        return {
            "added_lines": added,
            "removed_lines": removed,
            "changes_count": len(added) + len(removed)
        }

    def compare_policies(self, before_policy: dict[str, Any], after_policy: dict[str, Any]) -> dict[str, Any]:
        """Compares two policy configurations to identify differences in rules or permissions."""
        diffs = {}
        all_keys = set(before_policy.keys()).union(after_policy.keys())
        for key in all_keys:
            if key not in before_policy:
                diffs[key] = {"type": "added", "value": after_policy[key]}
            elif key not in after_policy:
                diffs[key] = {"type": "removed", "value": before_policy[key]}
            elif before_policy[key] != after_policy[key]:
                diffs[key] = {
                    "type": "modified",
                    "old_value": before_policy[key],
                    "new_value": after_policy[key]
                }
        return diffs


class TrendAnalyzer:
    """
    TrendAnalyzer computes direction, growth/decline patterns, moving averages,
    and spots anomalies across a timeline sequence of values.
    """

    def analyze_trend(self, data_points: Sequence[tuple[datetime, float]]) -> dict[str, Any]:
        """
        Analyzes a chronological list of (date, value) tuples.
        Computes growth/decline rate, moving average, detects anomalies, and generates summaries.
        """
        if not data_points:
            return {"status": "no_data"}

        # Sort chronologically
        sorted_points = sorted(data_points, key=lambda x: x[0])
        values = [p[1] for p in sorted_points]

        # 1. Growth / Decline
        start_val = values[0]
        end_val = values[-1]
        overall_diff = end_val - start_val
        overall_pct = (overall_diff / start_val * 100.0) if start_val != 0.0 else 0.0
        direction = "growth" if overall_diff > 0 else "decline" if overall_diff < 0 else "stable"

        # 2. Moving Average (window=3)
        moving_averages = []
        for i in range(len(values)):
            window = values[max(0, i-2):i+1]
            avg = sum(window) / len(window)
            moving_averages.append(avg)

        # 3. Anomaly Detection (Simple standard deviation threshold check)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        anomalies = []
        # Threshold: > 2 standard deviations away from mean
        if std_dev > 0.0:
            for i, val in enumerate(values):
                if abs(val - mean) > 1.5 * std_dev:
                    anomalies.append({
                        "date": sorted_points[i][0].isoformat(),
                        "value": val,
                        "deviation": round((val - mean) / std_dev, 2)
                    })

        # 4. Summary text
        summary = f"Values changed from {start_val} to {end_val}, indicating an overall {direction} of {round(overall_pct, 2)}%."

        return {
            "direction": direction,
            "overall_change": round(overall_diff, 4),
            "percentage_change": round(overall_pct, 2),
            "moving_averages": moving_averages,
            "anomalies": anomalies,
            "summary": summary
        }
