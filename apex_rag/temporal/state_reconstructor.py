from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from apex_rag.core.metrics.parser import MetricValueParser, UnitType
from apex_rag.ingestion.apex_storage import (
    ApexStorage,
    ChangeHistoryRow,
    NodeVersionRow,
)

# Backward-compatible alias — CausalEdgeRow is now KnowledgeEdgeRow
from apex_rag.ingestion.apex_storage import KnowledgeEdgeRow as CausalEdgeRow

logger = logging.getLogger("apex_rag.temporal.reconstructor")


class StateReconstructor:
    """
    StateReconstructor rebuilds the state of documents, graphs, or business records
    at any target point in history using NodeVersionRows, StateSnapshotRows, and CausalEdges.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self.storage = storage

    async def _get_nodes_as_of(self, doc_id: str, as_of: datetime) -> list[NodeVersionRow]:
        if not hasattr(self.storage, "get_nodes_as_of"):
            return []
        res = await self.storage.get_nodes_as_of(doc_id, as_of, tenant_context="default")
        return list(res) if res else []

    async def _get_state_snapshot(self, entity_id: str, as_of: datetime) -> Any:
        if not hasattr(self.storage, "get_state_snapshot"):
            return None
        res = await self.storage.get_state_snapshot(entity_id, as_of)
        return res

    async def _get_causal_edges(self, as_of: datetime) -> list[CausalEdgeRow]:
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(CausalEdgeRow).where(CausalEdgeRow.discovered_at <= as_of)
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def _get_change_history(self, entity_id: str, as_of: datetime) -> list[ChangeHistoryRow]:
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                from apex_rag.ingestion.apex_storage import ChangeHistoryRow

                stmt = (
                    select(ChangeHistoryRow)
                    .where(
                        ChangeHistoryRow.entity_id == entity_id,
                        ChangeHistoryRow.changed_at <= as_of,
                    )
                    .order_by(ChangeHistoryRow.changed_at.asc())
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def reconstruct_document_state(self, doc_id: str, as_of: datetime) -> str:
        """
        Reconstructs the full raw text content of a document as of the target datetime
        by concatenating all active node contents in order.

        Nodes are sorted by tree position (version_number, created_at) to preserve
        document structure. String-based node_id sorting (n1, n10, n2) is avoided.
        """
        # Fetch nodes effective as of the date
        nodes = await self._get_nodes_as_of(doc_id, as_of)
        if not nodes:
            return ""

        # Sort by tree position: version_number first, then created_at as tiebreaker
        # This preserves document structure better than string-sorting node_id
        nodes.sort(
            key=lambda n: (
                getattr(n, "version_number", 0) or 0,
                getattr(n, "created_at", datetime.min) or datetime.min,
            )
        )
        return "\n\n".join(getattr(node, "content", "") or "" for node in nodes)

    async def reconstruct_graph_state(self, doc_id: str, as_of: datetime) -> dict[str, Any]:
        """
        Reconstructs the graph state (nodes and active edges) for a document as of a target datetime.
        """
        # 1. Fetch active nodes
        nodes = await self._get_nodes_as_of(doc_id, as_of)
        node_ids = {n.node_id for n in nodes}

        # 2. Fetch causal edges discovered/valid at that time
        edge_rows = await self._get_causal_edges(as_of)

        # 3. Filter edges connecting only to active nodes
        active_edges = []
        for er in edge_rows:
            if er.source_node_id in node_ids and er.target_node_id in node_ids:
                active_edges.append(
                    {
                        "edge_id": er.edge_id,
                        "source_node_id": er.source_node_id,
                        "target_node_id": er.target_node_id,
                        "edge_type": er.edge_type,
                        "strength": er.strength,
                        "evidence": er.evidence,
                        "discovered_at": er.discovered_at.isoformat(),
                    }
                )

        return {
            "as_of": as_of.isoformat(),
            "nodes": [{"node_id": n.node_id, "version_number": n.version_number} for n in nodes],
            "edges": active_edges,
        }

    async def reconstruct_metrics(self, entity_id: str, as_of: datetime) -> dict[str, float]:
        """
        Reconstructs the active metrics/business record key-value pairs as of a target datetime.
        Usually relies on change_history or state_snapshots.
        """
        # Attempt to read from state_snapshots first
        snapshot = await self._get_state_snapshot(entity_id, as_of)
        if snapshot:
            data = getattr(snapshot, "snapshot_data", "")
            if data and isinstance(data, str):
                try:
                    return json.loads(data)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

        # Fallback to reconstructing from change_history
        change_rows = await self._get_change_history(entity_id, as_of)

        metrics = {}
        for change in change_rows:
            if change.new_value is not None:
                parsed = MetricValueParser.parse(change.new_value)
                if parsed.unit_type != UnitType.UNKNOWN:
                    metrics[change.field_name] = parsed.numeric_value
                else:
                    try:
                        metrics[change.field_name] = float(change.new_value)
                    except (ValueError, TypeError):
                        metrics[change.field_name] = 0.0
            else:
                metrics[change.field_name] = 0.0
        return metrics

    async def reconstruct_business_records(
        self, doc_id: str, as_of: datetime
    ) -> list[dict[str, Any]]:
        """
        Reconstructs business records contained within a document at a target point in time.
        """
        # Reconstruct by parsing nodes with tabular / list type or parsing content JSON if any
        nodes = await self._get_nodes_as_of(doc_id, as_of)
        records = []
        for node in nodes:
            # Simple content parser: if the node content contains key-value strings or json
            content = getattr(node, "content", "")
            if content and isinstance(content, str):
                if content.strip().startswith("{") and content.strip().endswith("}"):
                    with contextlib.suppress(json.JSONDecodeError):
                        records.append(json.loads(content))
                else:
                    # Add content as a general record
                    records.append(
                        {
                            "node_id": node.node_id,
                            "content": content,
                            "version_number": getattr(node, "version_number", 0),
                        }
                    )
        return records
