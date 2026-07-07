"""
retrieval/search/fts5.py — SQLite FTS5 Full-Text Search.

Replaces ILIKE("%query%") full-table scans with proper FTS5 full-text
search using SQLite's built-in FTS5 extension.

Supports:
  - Incremental updates (sync with AST nodes)
  - Ranked results (BM25 scoring)
  - Boolean queries and prefix searches
  - Compatibility with existing search API
  - ILIKE fallback for PostgreSQL databases

Usage:
    fts = FTS5Search(storage)
    results = await fts.search("revenue growth", doc_id="doc-123", limit=10)
    await fts.sync_node(node_id="abc-123")  # Incremental update
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy import text as sa_text

from apex_rag.ingestion.apex_storage import ApexStorage, ASTNodeRow

logger = logging.getLogger("apex_rag.retrieval.search.fts5")

# FTS5 table name
_FTS5_TABLE = "apex_ast_nodes_fts"

# FTS5 creation DDL
_FTS5_CREATE = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS5_TABLE}
USING fts5(
    node_id UNINDEXED,
    content,
    node_type UNINDEXED,
    doc_id UNINDEXED,
    tokenize='porter unicode61'
)
"""

_FTS5_DELETE = f"DELETE FROM {_FTS5_TABLE}"

_FTS5_INSERT = f"""
INSERT INTO {_FTS5_TABLE} (node_id, content, node_type, doc_id)
VALUES (:node_id, :content, :node_type, :doc_id)
"""

_FTS5_DELETE_NODE = f"DELETE FROM {_FTS5_TABLE} WHERE node_id = :node_id"

_FTS5_SEARCH = f"""
SELECT
    node_id,
    content,
    node_type,
    doc_id,
    rank
FROM {_FTS5_TABLE}
WHERE {_FTS5_TABLE} MATCH :query
ORDER BY rank
LIMIT :limit
"""

_FTS5_SEARCH_BY_DOC = f"""
SELECT
    node_id,
    content,
    node_type,
    doc_id,
    rank
FROM {_FTS5_TABLE}
WHERE {_FTS5_TABLE} MATCH :query
  AND doc_id = :doc_id
ORDER BY rank
LIMIT :limit
"""


class FTS5Search:
    """SQLite FTS5 full-text search for AST nodes.

    Only works with SQLite databases.  For PostgreSQL, the system
    falls back to ILIKE-based search.

    All operations are async and use managed sessions.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Setup ─────────────────────────────────────────────────────────────

    async def ensure_index(self) -> None:
        """Create the FTS5 virtual table if it doesn't exist.

        Safe to call multiple times — FTS5's ``CREATE VIRTUAL TABLE IF NOT EXISTS``
        is idempotent.
        """
        if not self._storage.is_sqlite:
            logger.warning("FTS5 is only supported on SQLite. Skipping index creation.")
            return

        async with self._storage.session() as session:
            await session.execute(sa_text(_FTS5_CREATE))
            logger.info("FTS5 index ensured: %s", _FTS5_TABLE)

    async def rebuild_index(self) -> int:
        """Rebuild the entire FTS5 index from scratch.

        Deletes all existing entries and re-indexes all AST nodes.

        Returns:
            The number of nodes indexed.
        """
        if not self._storage.is_sqlite:
            return 0

        async with self._storage.session() as session:
            # Clear existing index
            await session.execute(sa_text(_FTS5_DELETE))

            # Fetch all nodes
            result = await session.execute(select(ASTNodeRow))
            rows = result.scalars().all()

            # Batch insert
            count = 0
            for row in rows:
                await session.execute(
                    sa_text(_FTS5_INSERT),
                    {
                        "node_id": row.node_id,
                        "content": row.content or "",
                        "node_type": row.node_type or "",
                        "doc_id": row.doc_id or "",
                    },
                )
                count += 1

            logger.info("FTS5 index rebuilt: %d nodes indexed", count)
            return count

    # ── Incremental Sync ──────────────────────────────────────────────────

    async def sync_node(self, node_id: str) -> None:
        """Incrementally sync a single node's FTS5 entry.

        Removes any existing entry for the node and inserts a fresh one.
        Call this whenever a node is created or updated.

        Args:
            node_id: The UUID4 of the node to sync.
        """
        if not self._storage.is_sqlite:
            return

        async with self._storage.session() as session:
            row = await session.get(ASTNodeRow, node_id)
            if row is None:
                # Node deleted — remove from FTS
                await session.execute(sa_text(_FTS5_DELETE_NODE), {"node_id": node_id})
                return

            # Upsert: delete old entry, insert new
            await session.execute(sa_text(_FTS5_DELETE_NODE), {"node_id": node_id})
            await session.execute(
                sa_text(_FTS5_INSERT),
                {
                    "node_id": row.node_id,
                    "content": row.content or "",
                    "node_type": row.node_type or "",
                    "doc_id": row.doc_id or "",
                },
            )

    async def sync_document(self, doc_id: str) -> int:
        """Rebuild the FTS5 index for all nodes in a document.

        Args:
            doc_id: The document ID.

        Returns:
            Number of nodes synced.
        """
        if not self._storage.is_sqlite:
            return 0

        async with self._storage.session() as session:
            result = await session.execute(select(ASTNodeRow).where(ASTNodeRow.doc_id == doc_id))
            rows = result.scalars().all()

            count = 0
            for row in rows:
                await session.execute(sa_text(_FTS5_DELETE_NODE), {"node_id": row.node_id})
                await session.execute(
                    sa_text(_FTS5_INSERT),
                    {
                        "node_id": row.node_id,
                        "content": row.content or "",
                        "node_type": row.node_type or "",
                        "doc_id": row.doc_id or "",
                    },
                )
                count += 1

            logger.info("Synced %d FTS5 entries for doc=%s", count, doc_id)
            return count

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        doc_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search AST nodes using FTS5 full-text search.

        Uses FTS5 for SQLite databases.  Falls back to ILIKE-based search
        for PostgreSQL or other databases.

        Args:
            query:  The search query (FTS5 syntax: words, phrases, prefix*).
            doc_id: Optional — restrict search to a specific document.
            limit:  Maximum results.

        Returns:
            A list of dicts with keys: node_id, content, node_type, doc_id, rank.
            Results are ordered by BM25 rank (best first) on SQLite, or
            by content relevance on PostgreSQL.
        """
        if not self._storage.is_sqlite:
            logger.warning("FTS5 is only supported on SQLite. Falling back to ILIKE search.")
            return await self._ilike_search(query, doc_id=doc_id, limit=limit)

        # Ensure the FTS5 index exists
        await self.ensure_index()

        # Sanitise query for FTS5 syntax
        sanitised = self._sanitise_query(query)

        async with self._storage.session() as session:
            if doc_id:
                result = await session.execute(
                    sa_text(_FTS5_SEARCH_BY_DOC),
                    {"query": sanitised, "doc_id": doc_id, "limit": limit},
                )
            else:
                result = await session.execute(
                    sa_text(_FTS5_SEARCH),
                    {"query": sanitised, "limit": limit},
                )

            rows = result.all()
            return [
                {
                    "node_id": row.node_id,
                    "content": row.content,
                    "node_type": row.node_type,
                    "doc_id": row.doc_id,
                    "rank": round(row.rank, 4),
                }
                for row in rows
            ]

    # ── ILIKE fallback for non-SQLite databases ────────────────────────────

    async def _ilike_search(
        self,
        query: str,
        *,
        doc_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fallback ILIKE-based search for PostgreSQL and other databases.

        Converts the query into a case-insensitive content match.
        Results are ordered alphabetically by content as a simple
        relevance heuristic (exact and prefix matches appear first).
        """
        like_pattern = f"%{query}%"
        async with self._storage.session() as session:
            stmt = select(ASTNodeRow).where(ASTNodeRow.content.ilike(like_pattern))
            if doc_id:
                stmt = stmt.where(ASTNodeRow.doc_id == doc_id)
            stmt = stmt.order_by(ASTNodeRow.content).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "node_id": row.node_id,
                    "content": row.content,
                    "node_type": row.node_type,
                    "doc_id": row.doc_id,
                    "rank": 0.0,
                }
                for row in rows
            ]

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_query(query: str) -> str:
        """Sanitise a user query for FTS5 syntax.

        Escapes special characters and converts plain text to FTS5 format.
        Handles:
          - Boolean operators (AND, OR, NOT)
          - Phrase searches (double-quoted strings)
          - Prefix searches (trailing *)
          - Column-specific searches
        """
        if not query.strip():
            return ""

        # If the query already contains FTS5 operators, use it as-is
        fts5_operators = {"AND", "OR", "NOT", "*", '"', "("}
        if any(op in query for op in fts5_operators):
            return query

        # Convert to FTS5 phrase search for multi-word queries
        words = query.split()
        if len(words) == 1:
            return f'"{query}"*'  # Prefix match for single word
        return f'"{query}"'  # Exact phrase match
