from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from sqlalchemy import select
from apex_rag.ingestion.apex_storage import ApexStorage, NodeVersionRow, StateSnapshotRow, CausalEdgeRow
from apex_rag.models.unified_models import ASTNode, CausalEdge, EdgeType, NodeType

logger = logging.getLogger("apex_rag.temporal.reconstructor")

class StateReconstructor:
    """
    StateReconstructor rebuilds the state of documents, graphs, or business records
    at any target point in history using NodeVersionRows, StateSnapshotRows, and CausalEdges.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self.storage = storage

    async def reconstruct_document_state(self, doc_id: str, as_of: datetime) -> str:
        """
        Reconstructs the full raw text content of a document as of the target datetime
        by concatenating all active node contents in order.
        """
        # Fetch nodes effective as of the date
        nodes = await self.storage.get_nodes_as_of(doc_id, as_of)
        if not nodes:
            return ""

        # In order to reconstruct correctly, we can sort by node_id or revision/version details
        # For a simple structured reconstruction, we order by version_id/node_id
        nodes.sort(key=lambda n: n.node_id)
        return "\n\n".join(node.content for node in nodes)

    async def reconstruct_graph_state(self, doc_id: str, as_of: datetime) -> dict[str, Any]:
        """
        Reconstructs the graph state (nodes and active edges) for a document as of a target datetime.
        """
        # 1. Fetch active nodes
        nodes = await self.storage.get_nodes_as_of(doc_id, as_of)
        node_ids = {n.node_id for n in nodes}

        # 2. Fetch causal edges discovered/valid at that time
        async with self.storage.session() as session:
            stmt = select(CausalEdgeRow).where(
                CausalEdgeRow.discovered_at <= as_of
            )
            result = await session.execute(stmt)
            edge_rows = result.scalars().all()

        # 3. Filter edges connecting only to active nodes
        active_edges = []
        for er in edge_rows:
            if er.source_node_id in node_ids and er.target_node_id in node_ids:
                active_edges.append({
                    "edge_id": er.edge_id,
                    "source_node_id": er.source_node_id,
                    "target_node_id": er.target_node_id,
                    "edge_type": er.edge_type,
                    "strength": er.strength,
                    "evidence": er.evidence,
                    "discovered_at": er.discovered_at.isoformat()
                })

        return {
            "as_of": as_of.isoformat(),
            "nodes": [{"node_id": n.node_id, "version_number": n.version_number} for n in nodes],
            "edges": active_edges
        }

    async def reconstruct_metrics(self, entity_id: str, as_of: datetime) -> dict[str, float]:
        """
        Reconstructs the active metrics/business record key-value pairs as of a target datetime.
        Usually relies on change_history or state_snapshots.
        """
        # Attempt to read from state_snapshots first
        snapshot = await self.storage.get_state_snapshot(entity_id, as_of)
        if snapshot:
            try:
                return json.loads(snapshot.snapshot_data)
            except json.JSONDecodeError:
                pass

        # Fallback to reconstructing from change_history
        # Fetch all changes up to as_of
        async with self.storage.session() as session:
            from apex_rag.ingestion.apex_storage import ChangeHistoryRow
            stmt = select(ChangeHistoryRow).where(
                ChangeHistoryRow.entity_id == entity_id,
                ChangeHistoryRow.changed_at <= as_of
            ).order_by(ChangeHistoryRow.changed_at.asc())
            result = await session.execute(stmt)
            change_rows = result.scalars().all()

        metrics = {}
        for change in change_rows:
            try:
                metrics[change.field_name] = float(change.new_value) if change.new_value is not None else 0.0
            except ValueError:
                metrics[change.field_name] = change.new_value
        return metrics

    async def reconstruct_business_records(self, doc_id: str, as_of: datetime) -> list[dict[str, Any]]:
        """
        Reconstructs business records contained within a document at a target point in time.
        """
        # Reconstruct by parsing nodes with tabular / list type or parsing content JSON if any
        nodes = await self.storage.get_nodes_as_of(doc_id, as_of)
        records = []
        for node in nodes:
            # Simple content parser: if the node content contains key-value strings or json
            if node.content.strip().startswith("{") and node.content.strip().endswith("}"):
                try:
                    records.append(json.loads(node.content))
                except json.JSONDecodeError:
                    pass
            else:
                # Add content as a general record
                records.append({
                    "node_id": node.node_id,
                    "content": node.content,
                    "version_number": node.version_number
                })
        return records
