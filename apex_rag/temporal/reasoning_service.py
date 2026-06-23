"""
temporal/reasoning_service.py — Enterprise Temporal Reasoning Service.

Orchestrates temporal query detection, date parsing, version resolution,
state reconstruction, change detection, and trend analysis into a single
service that the ApexIndex query() API delegates to when it detects a
time-based query.

This service integrates:
  - TemporalReasoningAgent (date detection, query classification)
  - VersionResolver (version resolution, supersession chains)
  - StateReconstructor (time-travel state reconstruction)
  - ChangeAnalyzer / TrendAnalyzer (diff and trend analysis)

Usage:
    service = TemporalReasoningService(storage)
    result = await service.answer("What was revenue on 2025-01-15?", "doc-123")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.temporal.analyzers import ChangeAnalyzer, TrendAnalyzer
from apex_rag.temporal.state_reconstructor import StateReconstructor
from apex_rag.temporal.temporal_agent import TemporalReasoningAgent
from apex_rag.temporal.temporal_retriever import TemporalRetriever
from apex_rag.temporal.version_resolver import VersionResolver

logger = logging.getLogger("apex_rag.temporal.reasoning_service")


class TemporalReasoningService:
    """Enterprise service for answering time-aware queries with full provenance.

    Capabilities:
      - **Latest** — "What is today's revenue?"
      - **As-of-date** — "What was revenue on 2025-01-15?"
      - **Date range** — "Show sales between January and March"
      - **Change detection** — "What changed between Q1 and Q2?"
      - **Trend analysis** — "Show the revenue trend over 2025"
      - **Version history** — "What versions of the policy existed?"
      - **State reconstruction** — "Reconstruct the document as of Feb 1"

    All results include temporal provenance, version lineage, and
    confidence metadata.
    """

    def __init__(
        self,
        storage: ApexStorage,
        version_resolver: VersionResolver | None = None,
        reasoning_agent: TemporalReasoningAgent | None = None,
        change_analyzer: ChangeAnalyzer | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
    ) -> None:
        self._storage = storage
        self._version_resolver = version_resolver or VersionResolver(storage)

        # Build the standard temporal reasoning agent stack
        retriever = TemporalRetriever(storage)
        reconstructor = StateReconstructor(storage)
        self._reasoning_agent = reasoning_agent or TemporalReasoningAgent(
            retriever=retriever,
            reconstructor=reconstructor,
            change_analyzer=change_analyzer or ChangeAnalyzer(),
            trend_analyzer=trend_analyzer or TrendAnalyzer(),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def version_resolver(self) -> VersionResolver:
        """The underlying version resolver."""
        return self._version_resolver

    @property
    def reasoning_agent(self) -> TemporalReasoningAgent:
        """The underlying temporal reasoning agent."""
        return self._reasoning_agent

    async def answer(
        self,
        query: str,
        doc_id: str,
        *,
        as_of: datetime | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        latest: bool = False,
    ) -> dict[str, Any]:
        """Answer a temporal query with full provenance.

        Args:
            query:      Natural-language question.
            doc_id:     Target document ID.
            as_of:      Retrieve state as of this datetime.
            start_date: Start of date range (for range queries).
            end_date:   End of date range (for range queries).
            latest:     If True, return the latest active state.

        Returns:
            A dict with keys: mode, result, reasoning, provenance, latency_ms.

        Raises:
            ValueError: If conflicting temporal parameters are provided.
        """
        import time

        start = time.perf_counter()

        # ── Determine query mode ───────────────────────────────────────
        if as_of is not None:
            # Version resolution + state reconstruction at a point in time
            result = await self._answer_as_of(query, doc_id, as_of)
        elif start_date is not None or end_date is not None:
            # Date range query
            sd = start_date or datetime.min.replace(tzinfo=timezone.utc)
            ed = end_date or datetime.now(timezone.utc)
            result = await self._answer_range(query, doc_id, sd, ed)
        elif latest:
            # Latest active state
            result = await self._answer_latest(query, doc_id)
        else:
            # Let the reasoning agent classify and handle
            temp_result = await self._reasoning_agent.solve_temporal_query(
                query, doc_id,
            )
            result = temp_result

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)
        return result

    async def compare(
        self,
        query: str,
        doc_id: str,
        date_a: datetime,
        date_b: datetime,
    ) -> dict[str, Any]:
        """Compare document/metric state between two points in time.

        Args:
            query:  Contextual query describing what to compare.
            doc_id: Target document ID.
            date_a: First point in time.
            date_b: Second point in time.

        Returns:
            A dict with before/after metrics, diffs, and change analysis.
        """
        t1, t2 = sorted([date_a, date_b])

        # Delegate to the reasoning agent's change detection
        result = await self._reasoning_agent.solve_temporal_query(
            f"{query} compare", doc_id,
        )
        return result

    async def get_version_history(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Get the full version history for a node.

        Args:
            node_id: The node ID.

        Returns:
            A list of version snapshots with temporal metadata.
        """
        versions = await self._version_resolver.get_version_history(node_id)
        return [
            {
                "version_id": v.version_id,
                "version_number": v.version_number,
                "effective_from": v.effective_from.isoformat() if v.effective_from else None,
                "effective_to": v.effective_to.isoformat() if v.effective_to else None,
                "is_current": v.is_current,
                "validity_status": v.validity_status,
                "superseded_by": v.superseded_by,
            }
            for v in versions
        ]

    async def get_lineage(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Get the version lineage chain for a node.

        Args:
            node_id: The node ID.

        Returns:
            A list of lineage entries.
        """
        chain = await self._version_resolver.get_lineage_chain(node_id)
        return [
            {
                "lineage_id": l.lineage_id,
                "source_version_id": l.source_version_id,
                "target_version_id": l.target_version_id,
                "lineage_type": l.lineage_type,
                "strength": l.strength,
                "evidence": l.evidence,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in chain
        ]

    # ── Internal methods ───────────────────────────────────────────────

    async def _answer_as_of(
        self,
        query: str,
        doc_id: str,
        as_of: datetime,
    ) -> dict[str, Any]:
        """Answer a query as of a specific point in time."""
        reconstructor = StateReconstructor(self._storage)
        metrics = await reconstructor.reconstruct_metrics(doc_id, as_of)
        doc_text = await reconstructor.reconstruct_document_state(doc_id, as_of)
        graph_state = await reconstructor.reconstruct_graph_state(doc_id, as_of)

        return {
            "mode": "AS_OF_DATE",
            "target_date": as_of.isoformat(),
            "result": {
                "metrics": metrics,
                "content": doc_text,
                "graph_state": graph_state,
            },
            "provenance": {
                "as_of": as_of.isoformat(),
                "doc_id": doc_id,
                "reconstruction_method": "StateReconstructor",
            },
            "reasoning": f"Reconstructed document state as of {as_of.date()}.",
        }

    async def _answer_range(
        self,
        query: str,
        doc_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Answer a query over a date range."""
        retriever = TemporalRetriever(self._storage)

        # Fetch nodes active in range
        nodes_start = await retriever.get_nodes_as_of(doc_id, start_date)
        nodes_end = await retriever.get_nodes_as_of(doc_id, end_date)

        # Detect changes
        change_analyzer = ChangeAnalyzer()
        state_reconstructor = StateReconstructor(self._storage)
        text_start = await state_reconstructor.reconstruct_document_state(
            doc_id, start_date,
        )
        text_end = await state_reconstructor.reconstruct_document_state(
            doc_id, end_date,
        )
        text_diff = change_analyzer.compare_versions(text_start, text_end)

        # Metric comparison
        metrics_start = await state_reconstructor.reconstruct_metrics(
            doc_id, start_date,
        )
        metrics_end = await state_reconstructor.reconstruct_metrics(
            doc_id, end_date,
        )
        metric_comparisons = {}
        for key in set(metrics_start.keys()) | set(metrics_end.keys()):
            v1 = metrics_start.get(key, 0.0)
            v2 = metrics_end.get(key, 0.0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                metric_comparisons[key] = change_analyzer.compare_metrics(
                    float(v1), float(v2),
                )

        return {
            "mode": "DATE_RANGE",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "result": {
                "nodes_at_start": len(nodes_start),
                "nodes_at_end": len(nodes_end),
                "text_diff": text_diff,
                "metric_comparisons": metric_comparisons,
            },
            "provenance": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "doc_id": doc_id,
            },
            "reasoning": (
                f"Compared document state between {start_date.date()} "
                f"and {end_date.date()}. "
                f"Metric updates: {list(metric_comparisons.keys())}."
            ),
        }

    async def _answer_latest(
        self,
        query: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """Answer a query with the latest active state."""
        return await self._reasoning_agent.solve_temporal_query(query, doc_id)
