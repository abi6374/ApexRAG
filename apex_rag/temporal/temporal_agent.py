from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer

logger = logging.getLogger("apex_rag.temporal.agent")

class TemporalReasoningAgent:
    """
    TemporalReasoningAgent identifies temporal/time-based questions, parses date references,
    coordinates historical retrieval/reconstruction, analyzes changes or trends,
    and structures logical timeline reasoning responses.
    """

    def __init__(
        self,
        retriever: TemporalRetriever,
        reconstructor: StateReconstructor,
        change_analyzer: ChangeAnalyzer | None = None,
        trend_analyzer: TrendAnalyzer | None = None
    ) -> None:
        self.retriever = retriever
        self.reconstructor = reconstructor
        self.change_analyzer = change_analyzer or ChangeAnalyzer()
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()

    def detect_time_query(self, query: str) -> bool:
        """Determines if the user's query asks about historical versions, timelines, trends, or specific dates."""
        keywords = [
            r"\btoday\b", r"\byesterday\b", r"\bhistory\b", r"\btrend\b", r"\bchange\b", r"\bcompare\b",
            r"\bversion\b", r"\bactive\b", r"\bvalid\b", r"\beffective\b", r"\bbetween\b", r"\bfrom\b",
            r"\bto\b", r"\bq[1-4]\b", r"\bjanuary\b", r"\bfebruary\b", r"\bmarch\b", r"\bapril\b", r"\bmay\b",
            r"\bjune\b", r"\bjuly\b", r"\baugust\b", r"\bseptember\b", r"\boctober\b", r"\bnovember\b", r"\bdecember\b",
            r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{4}\b", r"\bwas\b", r"\bwere\b", r"\bhad\b", r"\bonce\b", r"\bold\b"
        ]
        combined = "|".join(keywords)
        return bool(re.search(combined, query, re.IGNORECASE))

    def parse_dates(self, query: str) -> list[datetime]:
        """Parses calendar dates or years from text. Fallback to current year if format permits."""
        dates = []
        # Match YYYY-MM-DD
        iso_matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        for iso_str in iso_matches:
            try:
                dates.append(datetime.strptime(iso_str, "%Y-%m-%d").replace(tzinfo=timezone.utc))
            except ValueError:
                pass

        # Match Year YYYY (2020-2030)
        year_matches = re.findall(r"\b(20[23]\d)\b", query)
        for yr_str in year_matches:
            # Skip if already matched as part of ISO
            if any(yr_str in m for m in iso_matches):
                continue
            dates.append(datetime(int(yr_str), 1, 1, tzinfo=timezone.utc))

        # Month words mapping (simplistic fallback parser for target months in current/2025 year)
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        for month_name, month_num in months.items():
            if re.search(rf"\b{month_name}\b", query, re.IGNORECASE):
                # If "Jan 2025" or similar year is matched in query, parse it.
                year_match = re.search(rf"\b{month_name}\s+(20\d{{2}})\b", query, re.IGNORECASE)
                year = int(year_match.group(1)) if year_match else 2025
                dates.append(datetime(year, month_num, 1, tzinfo=timezone.utc))

        # Quarters (Q1, Q2, etc.)
        q_matches = re.findall(r"\bq([1-4])\b", query, re.IGNORECASE)
        for q_str in q_matches:
            q = int(q_str)
            month = 1 if q == 1 else 4 if q == 2 else 7 if q == 3 else 10
            dates.append(datetime(2025, month, 1, tzinfo=timezone.utc))

        return dates

    async def solve_temporal_query(self, query: str, doc_id: str) -> dict[str, Any]:
        """
        Coordinates historical state retrieval and comparison logic.
        Returns a structured response containing facts, changes, or timeline analyses.
        """
        dates = self.parse_dates(query)
        logger.info("Parsed dates for temporal query: %s", dates)

        # 1. Trend Analysis: "Show sales trend..."
        if "trend" in query.lower() or "growth" in query.lower():
            # Gather timeline events or metric versions
            events = await self.retriever.storage.get_timeline_events(doc_id)
            if events:
                points = [(e.event_date, e.value if e.value is not None else 0.0) for e in events]
                trend = self.trend_analyzer.analyze_trend(points)
                return {
                    "mode": "TREND",
                    "result": trend,
                    "reasoning": f"Trend analysis performed over {len(events)} events: {trend['summary']}"
                }
            return {
                "mode": "TREND",
                "result": {},
                "reasoning": "Requested trend analysis but no timeline event records were found in the database."
            }

        # 2. Change / Compare: "Compare Q1 and Q2...", "What changed between..."
        if "change" in query.lower() or "compare" in query.lower() or "diff" in query.lower():
            if len(dates) >= 2:
                # Compare state at date[0] vs date[1]
                t1, t2 = sorted(dates)
                state1 = await self.reconstructor.reconstruct_metrics(doc_id, t1)
                state2 = await self.reconstructor.reconstruct_metrics(doc_id, t2)
                
                # Check for "Revenue" or key metrics
                metric_comparisons = {}
                for key in set(state1.keys()).union(state2.keys()):
                    val1 = state1.get(key, 0.0)
                    val2 = state2.get(key, 0.0)
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        metric_comparisons[key] = self.change_analyzer.compare_metrics(float(val1), float(val2))
                
                text1 = await self.reconstructor.reconstruct_document_state(doc_id, t1)
                text2 = await self.reconstructor.reconstruct_document_state(doc_id, t2)
                text_diff = self.change_analyzer.compare_versions(text1, text2)

                return {
                    "mode": "CHANGE_DETECTION",
                    "result": {
                        "metric_comparisons": metric_comparisons,
                        "text_diff": text_diff,
                        "date_before": t1.isoformat(),
                        "date_after": t2.isoformat()
                    },
                    "reasoning": f"Compared state between {t1.date()} and {t2.date()}. Metric updates: {list(metric_comparisons.keys())}."
                }

        # 3. As of specific Date: "What was revenue on 2025-01-10?"
        if dates:
            target_date = dates[0]
            metrics = await self.reconstructor.reconstruct_metrics(doc_id, target_date)
            doc_text = await self.reconstructor.reconstruct_document_state(doc_id, target_date)
            return {
                "mode": "AS_OF_DATE",
                "target_date": target_date.isoformat(),
                "result": {
                    "metrics": metrics,
                    "content": doc_text
                },
                "reasoning": f"Reconstructed historical document and metric states as of {target_date.date()}."
            }

        # 4. Latest Fallback: "What is today's revenue?"
        latest_nodes = await self.retriever.get_latest_nodes(doc_id)
        latest_metrics = await self.reconstructor.reconstruct_metrics(doc_id, datetime.now(timezone.utc))
        return {
            "mode": "LATEST",
            "result": {
                "metrics": latest_metrics,
                "nodes_count": len(latest_nodes)
            },
            "reasoning": "Returned the latest active current node state and active metrics values."
        }
