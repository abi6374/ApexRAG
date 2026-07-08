"""
agents/planner/temporal.py — Temporal Planner Agent.

Detects temporal scope in the query plan **before** navigation runs.
This is a **deterministic** stage — uses regex patterns and the
VersionResolver, no LLM calls.

Modes detected:
    - ``LATEST``: Query has no time constraints (default).
    - ``AS_OF``: Query targets a specific date (e.g. "as of 2025-01-10").
    - ``BETWEEN``: Query targets a date range (e.g. "between Q1 and Q2").
    - ``TREND``: Query asks about trends over time.

Usage:
    temporal_planner = TemporalPlannerAgent()
    plan = await temporal_planner.process(plan, context)
"""

from __future__ import annotations

import contextlib
import re
from datetime import datetime, timezone
from typing import Any

from apex_rag.agents.planner.models import EnrichedPlan, PlanningContext

# ── Temporal keyword patterns ──────────────────────────────────────────


_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_YEAR = re.compile(r"\b(20[23]\d)\b")
_DATE_QUARTER = re.compile(r"\bq([1-4])\b", re.IGNORECASE)

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_MONTH_PATTERN = re.compile(
    rf"\b({'|'.join(_MONTH_NAMES.keys())})\b", re.IGNORECASE
)

_TEMPORAL_KEYWORDS = re.compile(
    r"\b(as of|as at|effective|valid|history|trend|change|"
    r"compare|version|between|from|to|"
    r"today|yesterday|was|were|had|once|old)\b",
    re.IGNORECASE,
)


class TemporalPlannerAgent:
    """Detects temporal scope and applies version constraints to the plan.

    No LLM calls — uses regex pattern matching and date parsing.

    The detected ``temporal_mode`` and ``date_constraints`` are used by:
        - The Navigator (to filter nodes by temporal validity)
        - The TemporalRetriever (for version resolution)
        - The FreshnessScorer (for domain-specific decay)
    """

    def __init__(self) -> None:
        pass

    async def process(self, plan: EnrichedPlan, context: PlanningContext) -> EnrichedPlan:
        """Detect temporal scope in the plan.

        Args:
            plan:    The plan from RolePlannerAgent.
            context: Runtime context with doc_id.

        Returns:
            The enriched plan with temporal constraints.
        """
        # 1. Check if the query type or any sub-query is temporal
        query_text = " ".join(plan.sub_queries)
        query_type = plan.query_type

        if query_type == "TEMPORAL":
            plan.has_temporal_query = True

        if not plan.has_temporal_query and _TEMPORAL_KEYWORDS.search(query_text):
            plan.has_temporal_query = True

        if not plan.has_temporal_query:
            # No temporal component — nothing to do
            return plan

        # 2. Parse dates from the sub-queries
        dates = self._parse_dates(query_text)

        # 3. Determine temporal mode
        plan.date_constraints = dates
        plan.temporal_mode = self._detect_temporal_mode(
            query_text, dates, query_type, has_temporal_query=plan.has_temporal_query
        )

        # 4. Version filters based on detected mode
        plan.version_filters = self._build_version_filters(
            plan.temporal_mode, dates, context
        )

        return plan

    def _detect_temporal_mode(
        self,
        query_text: str,
        dates: list[datetime],
        query_type: str,
        has_temporal_query: bool = False,
    ) -> str | None:
        """Determine the temporal retrieval mode.

        Args:
            query_text:         Concatenated sub-queries.
            dates:              Parsed date objects.
            query_type:         The plan's query type.
            has_temporal_query: Whether the query was flagged as temporal.

        Returns:
            One of ``LATEST``, ``AS_OF``, ``BETWEEN``, ``TREND``, or ``None``.
        """
        query_lower = query_text.lower()

        if query_type == "TEMPORAL":
            if "trend" in query_lower or "growth" in query_lower:
                return "TREND"
            if len(dates) >= 2 or "between" in query_lower:
                return "BETWEEN"
            if dates:
                return "AS_OF"
            return "LATEST"

        if len(dates) >= 2 or "between" in query_lower:
            return "BETWEEN"
        if dates:
            return "AS_OF"

        return "LATEST" if has_temporal_query else None

    def _parse_dates(self, text: str) -> list[datetime]:
        """Parse dates from text using deterministic patterns.

        Args:
            text: The query text to parse.

        Returns:
            Sorted list of datetime objects.
        """
        dates: list[datetime] = []
        seen: set[str] = set()

        # ISO dates (YYYY-MM-DD)
        for match in _DATE_ISO.findall(text):
            if match not in seen:
                seen.add(match)
                with contextlib.suppress(ValueError):
                    dates.append(datetime.strptime(match, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    ))

        # Years (YYYY) — only if not part of ISO date
        year_matches = _DATE_YEAR.findall(text)
        for yr in year_matches:
            if yr not in seen:
                seen.add(yr)
                dates.append(datetime(int(yr), 1, 1, tzinfo=timezone.utc))

        # Month names
        for match in _DATE_MONTH_PATTERN.finditer(text):
            month_name = match.group(1).lower()
            month_num = _MONTH_NAMES[month_name]
            if month_name not in seen:
                seen.add(month_name)
                # Try to find a year adjacent to the month
                before = text[: match.start()].strip().split()[-1:] if match.start() > 0 else []
                after = text[match.end() :].strip().split()[:1] if match.end() < len(text) else []
                year_found = None
                for token in before + after:
                    try:
                        y = int(token)
                        if 2000 <= y <= 2099:
                            year_found = y
                            break
                    except ValueError:
                        pass
                dates.append(
                    datetime(year_found or 2025, month_num, 1, tzinfo=timezone.utc)
                )

        # Quarters (Q1, Q2, Q3, Q4)
        for match in _DATE_QUARTER.finditer(text):
            q = int(match.group(1))
            month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
            dates.append(datetime(2025, month, 1, tzinfo=timezone.utc))

        # Deduplicate and sort
        seen_dates: set[tuple[int, int, int]] = set()
        unique_dates: list[datetime] = []
        for d in sorted(dates):
            key = (d.year, d.month, d.day)
            if key not in seen_dates:
                seen_dates.add(key)
                unique_dates.append(d)

        return unique_dates

    def _build_version_filters(
        self,
        temporal_mode: str | None,
        dates: list[datetime],
        context: PlanningContext,
    ) -> dict[str, Any]:
        """Build version resolution filters based on temporal mode.

        Args:
            temporal_mode: The detected temporal mode.
            dates:         Parsed date constraints.
            context:       Runtime context.

        Returns:
            A dict of version filter parameters.
        """
        filters: dict[str, Any] = {
            "mode": temporal_mode,
            "domain": context.domain,
        }

        if temporal_mode == "AS_OF" and dates:
            filters["as_of"] = dates[0].isoformat()
        elif temporal_mode == "BETWEEN" and len(dates) >= 2:
            filters["from_date"] = dates[0].isoformat()
            filters["to_date"] = dates[1].isoformat()
        elif temporal_mode == "TREND":
            filters["include_trend"] = True

        return filters
