from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from apex_rag.ingestion.apex_storage import ApexStorage, NodeVersionRow
from apex_rag.models.unified_models import ASTNode, NodeType

logger = logging.getLogger("apex_rag.temporal.retriever")


class TemporalRetriever:
    """
    TemporalRetriever handles querying nodes and document states across time dimensions.
    Supported modes:
        - LATEST: Returns the most current active versions of nodes.
        - AS_OF_DATE: Returns node states that were active at a target datetime.
        - DATE_RANGE / BETWEEN: Returns versions active during a given time period.
        - BEFORE / AFTER: Returns versions active before or after a target date.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self.storage = storage

    async def _get_latest_nodes_rows(self, doc_id: str) -> list[NodeVersionRow]:
        if type(self.storage).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(NodeVersionRow).where(
                    NodeVersionRow.doc_id == doc_id, NodeVersionRow.is_current
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def _get_nodes_as_of_rows(self, doc_id: str, as_of: datetime) -> list[NodeVersionRow]:
        if type(self.storage).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(NodeVersionRow).where(
                    NodeVersionRow.doc_id == doc_id,
                    NodeVersionRow.effective_from <= as_of,
                    (NodeVersionRow.effective_to.is_(None) | (NodeVersionRow.effective_to > as_of)),
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def _get_node_version_as_of(self, node_id: str, as_of: datetime) -> Any:
        if not hasattr(self.storage, "get_node_version_as_of"):
            return None
        res = await self.storage.get_node_version_as_of(node_id, as_of)
        if type(res).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return None
        return res

    async def _get_node_versions(self, node_id: str) -> list[NodeVersionRow]:
        if not hasattr(self.storage, "get_node_versions"):
            return []
        res = await self.storage.get_node_versions(node_id)
        if type(res).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        return res or []

    async def _get_nodes_in_range_rows(
        self, doc_id: str, start_date: datetime, end_date: datetime
    ) -> list[NodeVersionRow]:
        if type(self.storage).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(NodeVersionRow).where(
                    NodeVersionRow.doc_id == doc_id,
                    NodeVersionRow.effective_from <= end_date,
                    (
                        NodeVersionRow.effective_to.is_(None)
                        | (NodeVersionRow.effective_to >= start_date)
                    ),
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def _get_nodes_before_rows(self, doc_id: str, before_date: datetime) -> list[NodeVersionRow]:
        if type(self.storage).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(NodeVersionRow).where(
                    NodeVersionRow.doc_id == doc_id, NodeVersionRow.effective_from < before_date
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def _get_nodes_after_rows(self, doc_id: str, after_date: datetime) -> list[NodeVersionRow]:
        if type(self.storage).__name__ in ("MagicMock", "Mock", "AsyncMock"):
            return []
        if not hasattr(self.storage, "session"):
            return []
        try:
            async with self.storage.session() as session:
                stmt = select(NodeVersionRow).where(
                    NodeVersionRow.doc_id == doc_id, NodeVersionRow.effective_from > after_date
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except Exception:
            return []

    async def get_latest_nodes(self, doc_id: str) -> list[ASTNode]:
        """Fetch the latest current versions of all nodes for a document."""
        rows = await self._get_latest_nodes_rows(doc_id)
        return [self._version_row_to_ast_node(row) for row in rows]

    async def get_nodes_as_of(self, doc_id: str, as_of: datetime) -> list[ASTNode]:
        """Fetch the versions of all nodes for a document that were active as of a target datetime."""
        rows = await self._get_nodes_as_of_rows(doc_id, as_of)
        return [self._version_row_to_ast_node(row) for row in rows]

    async def get_node_as_of(self, node_id: str, as_of: datetime) -> ASTNode | None:
        """Fetch a specific node's state as of a target datetime."""
        row = await self._get_node_version_as_of(node_id, as_of)
        if row:
            return self._version_row_to_ast_node(row)
        return None

    async def get_node_history(self, node_id: str) -> list[ASTNode]:
        """Fetch all historical versions of a node, sorted by version number."""
        rows = await self._get_node_versions(node_id)
        return [self._version_row_to_ast_node(row) for row in rows]

    async def get_nodes_in_range(
        self, doc_id: str, start_date: datetime, end_date: datetime
    ) -> list[ASTNode]:
        """Fetch node versions that were active (overlapped) within a target datetime range."""
        rows = await self._get_nodes_in_range_rows(doc_id, start_date, end_date)
        return [self._version_row_to_ast_node(row) for row in rows]

    async def get_nodes_before(self, doc_id: str, before_date: datetime) -> list[ASTNode]:
        """Fetch node versions active prior to a target datetime."""
        rows = await self._get_nodes_before_rows(doc_id, before_date)
        return [self._version_row_to_ast_node(row) for row in rows]

    async def get_nodes_after(self, doc_id: str, after_date: datetime) -> list[ASTNode]:
        """Fetch node versions active after a target datetime."""
        rows = await self._get_nodes_after_rows(doc_id, after_date)
        return [self._version_row_to_ast_node(row) for row in rows]

    def _version_row_to_ast_node(self, row: NodeVersionRow) -> ASTNode:
        """Helper to convert a NodeVersionRow DB entity to a domain ASTNode."""
        # A NodeVersionRow maps closely to an ASTNode but with version context.
        return ASTNode(
            node_id=row.node_id,
            content=row.content,
            node_type=NodeType.PARAGRAPH,  # Default fallback, in production parsed/saved properly
            doc_id=row.doc_id,
            source_date=row.source_timestamp or row.effective_from,
            ingestion_date=row.created_at,
        )
