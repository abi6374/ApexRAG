"""
apex_storage.py — Async SQLAlchemy storage layer for the unified ApexRAG models.

Stores :class:`ASTNode`, :class:`TemporalMetadata`, and :class:`CausalEdge`
objects in separate relational tables with proper foreign keys.

Supports both **SQLite** (development) and **PostgreSQL** (production).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    delete,
    event,
    select,
    text as sa_text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from apex_rag.models.unified_models import (
    ASTNode,
    CausalEdge,
    EdgeType,
    EvidencePacket,
    NodeType,
    TemporalMetadata,
)

logger = logging.getLogger("apex_rag.storage")


# ═══════════════════════════════════════════════════════════════
# ORM Base & Tables
# ═══════════════════════════════════════════════════════════════


class ApexBase(DeclarativeBase):
    """Shared declarative base for ApexRAG storage models."""


class ASTNodeRow(ApexBase):
    """SQL row for the unified :class:`ASTNode` model."""

    __tablename__ = "apex_ast_nodes"

    node_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    node_type: Mapped[str] = mapped_column(String(20), nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("apex_ast_nodes.node_id", ondelete="SET NULL"), nullable=True
    )
    children_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingestion_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    __table_args__ = (
        Index("ix_apex_nodes_doc", "doc_id"),
        Index("ix_apex_nodes_parent", "parent_id"),
        Index("ix_apex_nodes_type", "node_type"),
    )


class TemporalMetadataRow(ApexBase):
    """SQL row for the :class:`TemporalMetadata` model."""

    __tablename__ = "apex_temporal_metadata"

    node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingestion_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    freshness_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    decay_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.001)
    superseded_by: Mapped[str | None] = mapped_column(String(36), nullable=True)


class PageIndexEntryRow(ApexBase):
    """SQL row for the book-style page index.

    Maps a heading / keyword term to the ASTNode that covers it,
    including the page number.  Enables fast lookup of "which page
    does topic X appear on?".
    """

    __tablename__ = "apex_page_index"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    term: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (Index("ix_apex_pie_doc_term", "doc_id", "term"),)


class CausalEdgeRow(ApexBase):
    """SQL row for the :class:`CausalEdge` model."""

    __tablename__ = "apex_causal_edges"

    edge_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    edge_type: Mapped[str] = mapped_column(String(20), nullable=False)
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_apex_edges_source", "source_node_id"),
        Index("ix_apex_edges_target", "target_node_id"),
        Index("ix_apex_edges_type", "edge_type"),
    )


class QueryCacheRow(ApexBase):
    """Semantic cache for query results in the new ApexStorage schema.

    Uses a composite primary key of ``(query_hash, doc_id)`` so that
    the same query text across different documents doesn't collide.
    """

    __tablename__ = "apex_query_cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    node_ids: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════
# ApexStorage
# ═══════════════════════════════════════════════════════════════


class ApexStorage:
    """Async SQLAlchemy storage layer for the unified ApexRAG models.

    Usage::

        storage = await ApexStorage.create("sqlite+aiosqlite:///apex.db")

        # Save nodes
        await storage.save_nodes([node1, node2])

        # Retrieve
        nodes = await storage.get_nodes_by_doc("doc-123")
    """

    def __init__(
        self, engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in str(self._engine.url)

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        db_url: str = "sqlite+aiosqlite:///apex_rag.db",
        *,
        echo: bool = False,
    ) -> ApexStorage:
        """Async factory — creates the engine and ensures schema exists.

        Args:
            db_url: SQLAlchemy async URL.
                    SQLite:   ``sqlite+aiosqlite:///./apex_rag.db``
                    Postgres: ``postgresql+asyncpg://user:pass@host/db``
            echo:   Enable SQL query logging (dev only).

        Returns:
            A fully initialised :class:`ApexStorage` instance.
        """
        engine = create_async_engine(db_url, echo=echo)

        if db_url.startswith("sqlite"):

            @event.listens_for(engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn: Any, _: Any) -> None:
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
                dbapi_conn.execute("PRAGMA synchronous=NORMAL")
                dbapi_conn.execute("PRAGMA foreign_keys=ON")

        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        instance = cls(engine, session_factory)
        await instance._create_schema()
        logger.info(
            "ApexStorage ready (%s): %s",
            "SQLite" if db_url.startswith("sqlite") else "PostgreSQL",
            db_url.split("?")[0],
        )
        return instance

    async def _create_schema(self) -> None:
        """Create all tables if they don't already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(ApexBase.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop all ApexRAG tables — use with caution (tests only)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(ApexBase.metadata.drop_all)

    async def dispose(self) -> None:
        """Release all pooled connections."""
        if not hasattr(self, "_engine") or self._engine is None:
            return
        await self._engine.dispose()

    # ── Session context manager ────────────────────────────────────────────

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a managed async session with automatic rollback on error."""
        async with self._session_factory() as sess:
            try:
                yield sess
                await sess.commit()
            except Exception:
                await sess.rollback()
                raise

    # ── Node CRUD ──────────────────────────────────────────────────────────

    async def save_node(self, node: ASTNode, session: AsyncSession | None = None) -> None:
        """Persist a single AST node.

        Args:
            node:    The AST node to save.
            session: Optional existing session (creates one if omitted).
        """
        if session is not None:
            await self._save_node_single(session, node)
        else:
            async with self.session() as sess:
                await self._save_node_single(sess, node)

    async def save_nodes(
        self, nodes: list[ASTNode], session: AsyncSession | None = None
    ) -> None:
        """Persist multiple AST nodes in a single transaction.

        Args:
            nodes:   The AST nodes to save.
            session: Optional existing session.
        """
        if session is not None:
            for node in nodes:
                await self._save_node_single(session, node)
        else:
            async with self.session() as sess:
                for node in nodes:
                    await self._save_node_single(sess, node)

    async def _save_node_single(self, session: AsyncSession, node: ASTNode) -> None:
        """Map an ASTNode to a row and INSERT or UPDATE."""
        existing = await session.get(ASTNodeRow, node.node_id)
        if existing is not None:
            # Update
            existing.content = node.content
            node_type_str = node.node_type if isinstance(node.node_type, str) else node.node_type.value
            existing.node_type = node_type_str
            existing.depth = node.depth
            existing.parent_id = node.parent_id
            existing.children_json = json.dumps(node.children)
            existing.source_date = node.source_date
            existing.ingestion_date = node.ingestion_date
            existing.embedding_json = json.dumps(node.embedding)
            existing.page_number = node.page_number
        else:
            node_type_str = node.node_type if isinstance(node.node_type, str) else node.node_type.value
            row = ASTNodeRow(
                node_id=node.node_id,
                content=node.content,
                node_type=node_type_str,
                depth=node.depth,
                parent_id=node.parent_id,
                children_json=json.dumps(node.children),
                doc_id=node.doc_id,
                source_date=node.source_date,
                ingestion_date=node.ingestion_date,
                embedding_json=json.dumps(node.embedding),
                page_number=node.page_number,
            )
            session.add(row)

    async def get_node(self, node_id: str) -> ASTNode | None:
        """Fetch a single AST node by its ID.

        Args:
            node_id: The UUID4 string identifying the node.

        Returns:
            The :class:`ASTNode` if found, or ``None``.
        """
        async with self.session() as session:
            row = await session.get(ASTNodeRow, node_id)
            if row is None:
                return None
            return _row_to_ast_node(row)

    async def get_nodes_by_doc(
        self, doc_id: str, session: AsyncSession | None = None
    ) -> list[ASTNode]:
        """Fetch all AST nodes for a given document.

        Args:
            doc_id:  The document ID.
            session: Optional existing session.

        Returns:
            A list of :class:`ASTNode` objects.
        """
        if session is not None:
            return await self._get_nodes_by_doc(session, doc_id)

        async with self.session() as sess:
            return await self._get_nodes_by_doc(sess, doc_id)

    async def _get_nodes_by_doc(self, session: AsyncSession, doc_id: str) -> list[ASTNode]:
        stmt = select(ASTNodeRow).where(ASTNodeRow.doc_id == doc_id)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [_row_to_ast_node(r) for r in rows]

    async def get_all_nodes(self) -> list[ASTNode]:
        """Fetch all AST nodes across all documents.

        Returns:
            A list of all :class:`ASTNode` objects.
        """
        async with self.session() as session:
            result = await session.execute(select(ASTNodeRow))
            rows = result.scalars().all()
            return [_row_to_ast_node(r) for r in rows]

    async def get_all_nodes(self) -> list[ASTNode]:
        """Fetch all AST nodes across all documents.

        Returns:
            A list of all :class:`ASTNode` objects.
        """
        async with self.session() as session:
            result = await session.execute(select(ASTNodeRow))
            rows = result.scalars().all()
            return [_row_to_ast_node(r) for r in rows]

    async def get_ast_children(
        self, session: AsyncSession, parent_id: str | None, doc_id: str
    ) -> Sequence[ASTNodeRow]:
        """Fetch child nodes for a given parent and document.

        Args:
            session:   AsyncSession.
            parent_id: ID of the parent node (None for root nodes).
            doc_id:    Document ID.

        Returns:
            A sequence of :class:`ASTNodeRow` objects.
        """
        if parent_id is None:
            stmt = select(ASTNodeRow).where(
                ASTNodeRow.doc_id == doc_id, ASTNodeRow.parent_id.is_(None)
            )
        else:
            stmt = select(ASTNodeRow).where(
                ASTNodeRow.doc_id == doc_id, ASTNodeRow.parent_id == parent_id
            )
        result = await session.execute(stmt)
        return result.scalars().all()

    async def get_node_by_id(self, session: AsyncSession, node_id: str) -> ASTNodeRow | None:
        """Internal helper for fetching a row by ID."""
        return await session.get(ASTNodeRow, node_id)

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node by its ID.

        Args:
            node_id: The UUID4 string identifying the node.

        Returns:
            ``True`` if the node existed and was deleted.
        """
        async with self.session() as session:
            row = await session.get(ASTNodeRow, node_id)
            if row is None:
                return False
            await session.delete(row)
            return True

    async def count_nodes(self, doc_id: str | None = None) -> int:
        """Count nodes, optionally filtered by document.

        Args:
            doc_id: Optional document ID to filter by.

        Returns:
            The node count.
        """
        async with self.session() as session:
            if doc_id is None:
                result = await session.execute(sa_text("SELECT COUNT(*) FROM apex_ast_nodes"))
            else:
                result = await session.execute(
                    sa_text("SELECT COUNT(*) FROM apex_ast_nodes WHERE doc_id = :did"),
                    {"did": doc_id},
                )
            return result.scalar() or 0

    # ── Temporal Metadata CRUD ─────────────────────────────────────────────

    async def save_temporal_metadata(self, meta: TemporalMetadata) -> None:
        """Persist temporal metadata for a node.

        Args:
            meta: The :class:`TemporalMetadata` to save.
        """
        async with self.session() as session:
            existing = await session.get(TemporalMetadataRow, meta.node_id)
            if existing is not None:
                existing.source_date = meta.source_date
                existing.ingestion_date = meta.ingestion_date
                existing.freshness_score = meta.freshness_score
                existing.decay_rate = meta.decay_rate
                existing.superseded_by = meta.superseded_by
            else:
                row = TemporalMetadataRow(
                    node_id=meta.node_id,
                    source_date=meta.source_date,
                    ingestion_date=meta.ingestion_date,
                    freshness_score=meta.freshness_score,
                    decay_rate=meta.decay_rate,
                    superseded_by=meta.superseded_by,
                )
                session.add(row)

    async def get_temporal_metadata(self, node_id: str) -> TemporalMetadata | None:
        """Fetch temporal metadata for a node.

        Args:
            node_id: The node ID.

        Returns:
            :class:`TemporalMetadata` or ``None``.
        """
        async with self.session() as session:
            row = await session.get(TemporalMetadataRow, node_id)
            if row is None:
                return None
            return TemporalMetadata(
                node_id=row.node_id,
                source_date=row.source_date,
                ingestion_date=row.ingestion_date,
                freshness_score=row.freshness_score,
                decay_rate=row.decay_rate,
                superseded_by=row.superseded_by,
            )

    # ── Causal Edge CRUD ───────────────────────────────────────────────────

    async def save_causal_edge(self, edge: CausalEdge) -> None:
        """Persist a causal edge.

        Args:
            edge: The :class:`CausalEdge` to save.
        """
        async with self.session() as session:
            existing = await session.get(CausalEdgeRow, edge.edge_id)
            edge_type_str = edge.edge_type if isinstance(edge.edge_type, str) else edge.edge_type.value
            if existing is not None:
                existing.source_node_id = edge.source_node_id
                existing.target_node_id = edge.target_node_id
                existing.edge_type = edge_type_str
                existing.strength = edge.strength
                existing.evidence = edge.evidence
                existing.discovered_at = edge.discovered_at
            else:
                row = CausalEdgeRow(
                    edge_id=edge.edge_id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    edge_type=edge_type_str,
                    strength=edge.strength,
                    evidence=edge.evidence,
                    discovered_at=edge.discovered_at,
                )
                session.add(row)

    async def get_edges_for_node(
        self, node_id: str
    ) -> list[CausalEdge]:
        """Fetch all causal edges involving a given node (source or target).

        Args:
            node_id: The node ID.

        Returns:
            A list of :class:`CausalEdge` objects.
        """
        async with self.session() as session:
            stmt = select(CausalEdgeRow).where(
                (CausalEdgeRow.source_node_id == node_id)
                | (CausalEdgeRow.target_node_id == node_id)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_row_to_causal_edge(r) for r in rows]

    async def get_all_edges(self) -> list[CausalEdge]:
        """Fetch all causal edges across the graph.

        Returns:
            A list of all :class:`CausalEdge` objects.
        """
        async with self.session() as session:
            result = await session.execute(select(CausalEdgeRow))
            rows = result.scalars().all()
            return [_row_to_causal_edge(r) for r in rows]

    # ── Page Index CRUD ────────────────────────────────────────────────────

    async def save_page_index_entry(self, entry: dict[str, Any]) -> None:
        """Insert or update a page index entry.

        Args:
            entry: Dict with keys ``node_id``, ``doc_id``, ``term``,
                   and optionally ``page_number``.
        """
        async with self.session() as session:
            existing = await session.execute(
                select(PageIndexEntryRow).where(
                    PageIndexEntryRow.node_id == entry["node_id"],
                    PageIndexEntryRow.term == entry["term"],
                )
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                row.page_number = entry.get("page_number", row.page_number)
            else:
                row = PageIndexEntryRow(
                    node_id=entry["node_id"],
                    doc_id=entry["doc_id"],
                    term=entry["term"],
                    page_number=entry.get("page_number"),
                )
                session.add(row)

    async def save_page_index_entries(
        self, entries: list[dict[str, Any]]
    ) -> None:
        """Batch-insert page index entries."""
        async with self.session() as session:
            for entry in entries:
                row = PageIndexEntryRow(
                    node_id=entry["node_id"],
                    doc_id=entry["doc_id"],
                    term=entry["term"],
                    page_number=entry.get("page_number"),
                )
                session.add(row)

    async def get_page_index_entries(
        self, doc_id: str
    ) -> list[dict[str, Any]]:
        """Fetch all page index entries for a document, ordered by term."""
        async with self.session() as session:
            stmt = (
                select(PageIndexEntryRow)
                .where(PageIndexEntryRow.doc_id == doc_id)
                .order_by(PageIndexEntryRow.term)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "node_id": r.node_id,
                    "doc_id": r.doc_id,
                    "term": r.term,
                    "page_number": r.page_number,
                }
                for r in rows
            ]

    async def search_page_index(
        self, doc_id: str, query: str
    ) -> list[dict[str, Any]]:
        """Full-text search over page index terms (case-insensitive)."""
        async with self.session() as session:
            stmt = (
                select(PageIndexEntryRow)
                .where(
                    PageIndexEntryRow.doc_id == doc_id,
                    PageIndexEntryRow.term.ilike(f"%{query}%"),
                )
                .order_by(PageIndexEntryRow.term)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "node_id": r.node_id,
                    "doc_id": r.doc_id,
                    "term": r.term,
                    "page_number": r.page_number,
                }
                for r in rows
            ]

    # ── Semantic Cache ─────────────────────────────────────────────────────

    async def cache_query_result(
        self,
        query_hash: str,
        query_text: str,
        doc_id: str,
        node_ids: list[str],
    ) -> None:
        """Store a query result in the semantic cache (keyed by query_hash + doc_id)."""
        async with self.session() as session:
            existing = await session.get(QueryCacheRow, (query_hash, doc_id))
            if existing is not None:
                existing.query_text = query_text
                existing.node_ids = json.dumps(node_ids)
                existing.hit_count = existing.hit_count + 1
            else:
                row = QueryCacheRow(
                    query_hash=query_hash,
                    doc_id=doc_id,
                    query_text=query_text,
                    node_ids=json.dumps(node_ids),
                )
                session.add(row)

    async def get_cached_query(
        self, query_hash: str, doc_id: str
    ) -> dict[str, Any] | None:
        """Retrieve a cached query result by query_hash + doc_id."""
        async with self.session() as session:
            row = await session.get(QueryCacheRow, (query_hash, doc_id))
            if row is None:
                return None
            return {
                "query_hash": row.query_hash,
                "query": row.query_text,
                "doc_id": row.doc_id,
                "node_ids": json.loads(row.node_ids) if row.node_ids else [],
            }

    async def delete_cached_query(self, query_hash: str, doc_id: str) -> bool:
        """Remove a cached query entry by query_hash + doc_id."""
        async with self.session() as session:
            row = await session.get(QueryCacheRow, (query_hash, doc_id))
            if row is None:
                return False
            await session.delete(row)
            return True

    # ── Global Search ──────────────────────────────────────────────────────

    async def list_document_ids(self) -> list[str]:
        """List all document IDs that have stored AST nodes."""
        async with self.session() as session:
            stmt = select(ASTNodeRow.doc_id).distinct()
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def get_document_root_nodes(
        self, doc_id: str
    ) -> list[ASTNode]:
        """Fetch root-level nodes (parent_id is None) for a document."""
        async with self.session() as session:
            stmt = (
                select(ASTNodeRow)
                .where(
                    ASTNodeRow.doc_id == doc_id,
                    ASTNodeRow.parent_id.is_(None),
                )
                .order_by(ASTNodeRow.node_id)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_row_to_ast_node(r) for r in rows]

    async def search_nodes_global(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[ASTNode]:
        """Search all nodes across all documents using content similarity.

        Performs a simple case-insensitive LIKE search on content and
        node_type for now.  In production this would use FTS5 or vector search.

        Args:
            query: Search string.
            limit: Maximum results.

        Returns:
            Matching ASTNode objects.
        """
        async with self.session() as session:
            stmt = (
                select(ASTNodeRow)
                .where(ASTNodeRow.content.ilike(f"%{query}%"))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_row_to_ast_node(r) for r in rows]

    async def get_document_stats(self, doc_id: str) -> dict[str, Any]:
        """Return aggregate statistics for a document."""
        async with self.session() as session:
            stmt = select(ASTNodeRow).where(ASTNodeRow.doc_id == doc_id)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            nodes = [_row_to_ast_node(r) for r in rows]
            max_depth = max((n.depth for n in nodes), default=0)
            leaf_count = sum(1 for n in nodes if not n.children)
            return {
                "doc_id": doc_id,
                "total_nodes": len(nodes),
                "max_depth": max_depth,
                "leaf_count": leaf_count,
            }

    async def delete_document(self, doc_id: str) -> int:
        """Delete all nodes and page index entries for a document."""
        async with self.session() as session:
            # Delete page index entries first
            await session.execute(
                sa_text("DELETE FROM apex_page_index WHERE doc_id = :did"),
                {"did": doc_id},
            )
            # Delete causal edges for nodes in this doc
            await session.execute(
                sa_text(
                    "DELETE FROM apex_causal_edges WHERE source_node_id IN "
                    "(SELECT node_id FROM apex_ast_nodes WHERE doc_id = :did)"
                ),
                {"did": doc_id},
            )
            # Delete temporal metadata
            await session.execute(
                sa_text(
                    "DELETE FROM apex_temporal_metadata WHERE node_id IN "
                    "(SELECT node_id FROM apex_ast_nodes WHERE doc_id = :did)"
                ),
                {"did": doc_id},
            )
            # Delete nodes
            result = await session.execute(
                delete(ASTNodeRow).where(ASTNodeRow.doc_id == doc_id)
            )
            return result.rowcount


# ═══════════════════════════════════════════════════════════════
# Row → Model mappers
# ═══════════════════════════════════════════════════════════════


def _row_to_ast_node(row: ASTNodeRow) -> ASTNode:
    """Convert a database row to an :class:`ASTNode`."""
    children = json.loads(row.children_json) if row.children_json else []
    embedding = json.loads(row.embedding_json) if row.embedding_json else []
    return ASTNode(
        node_id=row.node_id,
        content=row.content,
        node_type=NodeType(row.node_type),
        depth=row.depth,
        parent_id=row.parent_id,
        children=children,
        doc_id=row.doc_id,
        source_date=row.source_date,
        ingestion_date=row.ingestion_date,
        embedding=embedding,
        page_number=row.page_number,
    )


def _row_to_causal_edge(row: CausalEdgeRow) -> CausalEdge:
    """Convert a database row to a :class:`CausalEdge`."""
    return CausalEdge(
        edge_id=row.edge_id,
        source_node_id=row.source_node_id,
        target_node_id=row.target_node_id,
        edge_type=EdgeType(row.edge_type),
        strength=row.strength,
        evidence=row.evidence,
        discovered_at=row.discovered_at,
    )
