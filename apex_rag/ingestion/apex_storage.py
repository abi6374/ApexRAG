"""
apex_storage.py — Async SQLAlchemy storage layer for the unified ApexRAG models.

Stores :class:`ASTNode`, :class:`TemporalMetadata`, and :class:`CausalEdge`
objects in separate relational tables with proper foreign keys.

Supports both **SQLite** (development) and **PostgreSQL** (production).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    delete,
    event,
    or_,
    select,
)
from sqlalchemy import (
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
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, default="default"
    )
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
        Index("ix_apex_nodes_tenant", "tenant_id"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approval_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    previous_version: Mapped[str | None] = mapped_column(String(36), nullable=True)
    validity_status: Mapped[str] = mapped_column(String(20), nullable=False, default="ACTIVE")


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


class NodeVersionRow(ApexBase):
    """SQL row representing a specific immutable version of an AST Node.

    Every change creates a new row (INSERT-only).  No UPDATEs modify
    historical data.  Foreign keys ensure referential integrity.

    Attributes:
        version_id:         Primary key (UUID4).
        node_id:            FK to the AST node being versioned.
        content:            The full node content at this version.
        content_hash:       SHA-256 hash of content for integrity checks.
        created_at:         When this version was created.
        updated_at:         Last update timestamp.
        effective_from:     When this version became active.
        effective_to:       When this version was superseded (NULL = current).
        version_number:     Monotonically increasing version counter.
        revision_number:    Optional revision within the same version.
        source_timestamp:   When the source was authored.
        is_current:         True if this is the latest version.
        superseded_by:      FK to the version that replaced this one.
        previous_version:   FK to the immediately preceding version.
        validity_status:    ACTIVE, PENDING, EXPIRED, SUPERSEDED, DRAFT, ARCHIVED.
        doc_id:             Document ID this node belongs to.
        tenant_id:          Tenant isolation boundary.
    """

    __tablename__ = "node_versions"

    version_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    superseded_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("node_versions.version_id", ondelete="SET NULL"),
        nullable=True,
    )
    previous_version: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("node_versions.version_id", ondelete="SET NULL"),
        nullable=True,
    )
    validity_status: Mapped[str] = mapped_column(String(20), nullable=False, default="ACTIVE")
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    __table_args__ = (
        Index("ix_node_versions_node_id", "node_id"),
        Index("ix_node_versions_doc", "doc_id"),
        Index("ix_node_versions_tenant", "tenant_id"),
        Index("ix_node_versions_current", "node_id", "is_current"),
        Index("ix_node_versions_effective", "node_id", "effective_from", "effective_to"),
    )


class TemporalNodeRow(ApexBase):
    """SQL row storing node validity periods and temporal attributes."""

    __tablename__ = "temporal_nodes"

    node_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    superseded_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    previous_version: Mapped[str | None] = mapped_column(String(36), nullable=True)


class AuditLogRow(ApexBase):
    """SQL row for enterprise RBAC and historical query audit logging."""

    __tablename__ = "audit_logs"

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    before_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    after_state: Mapped[str | None] = mapped_column(Text, nullable=True)


class ChangeHistoryRow(ApexBase):
    """SQL row tracking field-level changes across entities over time."""

    __tablename__ = "change_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    changed_by: Mapped[str] = mapped_column(String(255), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)


class TimelineEventRow(ApexBase):
    """SQL row tracking time-series numeric events (e.g. sales date and revenue amount)."""

    __tablename__ = "timeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    value: Mapped[float | None] = mapped_column(Float, nullable=True)


class RolePermissionRow(ApexBase):
    """SQL row defining role-level access permissions."""

    __tablename__ = "role_permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    is_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class FieldPermissionRow(ApexBase):
    """SQL row defining field-level access permissions/visibility rules."""

    __tablename__ = "field_permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class CustomRuleRow(ApexBase):
    """SQL row defining a custom policy rule with dynamic execution logic."""

    __tablename__ = "custom_rules"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    rule_type: Mapped[str] = mapped_column(String(50), nullable=False, default="expression")
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class RuleAssignmentRow(ApexBase):
    """SQL row assigning custom rules to roles or users."""

    __tablename__ = "rule_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    rule_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    is_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class StateSnapshotRow(ApexBase):
    """SQL row storing complete document/graph state snapshots at a point in time."""

    __tablename__ = "state_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    snapshot_data: Mapped[str] = mapped_column(Text, nullable=False)


class VersionLineageRow(ApexBase):
    """SQL row tracking version lineage chains across node versions.

    Each row records a directed relationship between two versions
    (e.g. version A SUPERSEDES version B), forming a DAG that
    enables full version ancestry traversal.
    """

    __tablename__ = "version_lineage"

    lineage_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, default="default"
    )
    source_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    target_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lineage_type: Mapped[str] = mapped_column(String(30), nullable=False, default="VERSION_OF")
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_vl_node", "node_id"),
        Index("ix_vl_source", "source_version_id"),
        Index("ix_vl_target", "target_version_id"),
        Index("ix_vl_type", "lineage_type"),
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
        """Create all tables if they don't already exist.

        Handles SQLite's lack of ``IF NOT EXISTS`` for indexes by
        catching and ignoring duplicate index errors PER TABLE.
        This prevents a single failing index from rolling back the
        creation of all other tables (which ``create_all`` does when
        run within a single transaction).

        Strategy:
          1. Fast path — try ``create_all`` once (succeeds on fresh DB).
          2. Fallback — if step 1 fails, create each table individually
             so a duplicate-index error on one table doesn't cascade.
        """
        # Fast path: single create_all for fresh databases
        async with self._engine.begin() as conn:
            try:
                await conn.run_sync(ApexBase.metadata.create_all)
                return
            except Exception:
                logger.warning(
                    "Fast-path create_all failed (likely duplicate index). "
                    "Falling back to per-table creation..."
                )

        # Fallback: create each table individually so a failure on one
        # table's indexes doesn't roll back the others.
        for table in ApexBase.metadata.sorted_tables:
            try:
                async with self._engine.begin() as conn:
                    await conn.run_sync(table.create, checkfirst=True)
            except Exception as exc:
                logger.warning(
                    "Table '%s' already exists or index failed: %s",
                    table.name,
                    exc,
                )

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

    @staticmethod
    def _get_table_name(model_class: type[ApexBase]) -> str:
        """Get the table name for a given ORM model class."""
        return model_class.__tablename__

    async def save_node(
        self,
        node: ASTNode,
        session: AsyncSession | None = None,
        *,
        tenant_context: str | None = None,
    ) -> None:
        """Persist a single AST node.

        Args:
            node:    The AST node to save.
            session: Optional existing session (creates one if omitted).
            tenant_context: Required tenant ID for multi-tenant isolation.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for save_node. "
                "All storage operations require a tenant context."
            )
        from apex_rag.enterprise.auth.tenant_validator import TenantIsolationValidator

        validator = TenantIsolationValidator(self)
        await validator.assert_tenant_write_access(tenant_context, self._get_table_name(ASTNodeRow))
        if session is not None:
            await self._save_node_single(session, node, tenant_context=tenant_context)
        else:
            async with self.session() as sess:
                await self._save_node_single(sess, node, tenant_context=tenant_context)

    async def save_nodes(
        self,
        nodes: list[ASTNode],
        session: AsyncSession | None = None,
        *,
        tenant_context: str | None = None,
    ) -> None:
        """Persist multiple AST nodes in a single transaction.

        Args:
            nodes:   The AST nodes to save.
            session: Optional existing session.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for save_nodes. "
                "All storage operations require a tenant context."
            )
        from apex_rag.enterprise.auth.tenant_validator import TenantIsolationValidator

        validator = TenantIsolationValidator(self)
        await validator.assert_tenant_write_access(tenant_context, self._get_table_name(ASTNodeRow))
        if session is not None:
            for node in nodes:
                await self._save_node_single(session, node)
        else:
            async with self.session() as sess:
                for node in nodes:
                    await self._save_node_single(sess, node)

    async def _save_node_single(
        self,
        session: AsyncSession,
        node: ASTNode,
        *,
        tenant_context: str | None = None,
    ) -> None:
        """Map an ASTNode to a row and INSERT or UPDATE.

        Args:
            session:         The active database session.
            node:            The AST node to persist.
            tenant_context:  Required tenant ID — used to set the tenant_id column.
        """
        existing = await session.get(ASTNodeRow, node.node_id)
        if existing is not None:
            # Update
            existing.content = node.content
            node_type_str = (
                node.node_type if isinstance(node.node_type, str) else node.node_type.value
            )
            existing.node_type = node_type_str
            existing.depth = node.depth
            existing.parent_id = node.parent_id
            existing.children_json = json.dumps(node.children)
            existing.source_date = node.source_date
            existing.ingestion_date = node.ingestion_date
            existing.embedding_json = json.dumps(node.embedding)
            existing.page_number = node.page_number
            if tenant_context:
                existing.tenant_id = tenant_context
        else:
            node_type_str = (
                node.node_type if isinstance(node.node_type, str) else node.node_type.value
            )
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
                tenant_id=tenant_context or "default",
            )
            session.add(row)

    async def get_node(self, node_id: str, *, tenant_context: str | None = None) -> ASTNode | None:
        """Fetch a single AST node by its ID.

        Args:
            node_id: The UUID4 string identifying the node.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            The :class:`ASTNode` if found, or ``None``.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for get_node. "
                "All storage operations require a tenant context."
            )
        from apex_rag.enterprise.auth.tenant_validator import TenantIsolationValidator

        validator = TenantIsolationValidator(self)
        await validator.assert_tenant_read_access(tenant_context, self._get_table_name(ASTNodeRow))
        async with self.session() as session:
            row = await session.get(ASTNodeRow, node_id)
            if row is None:
                return None
            # Verify tenant ownership
            if row.tenant_id and row.tenant_id != tenant_context:
                raise PermissionError(
                    f"Node {node_id} belongs to tenant '{row.tenant_id}', "
                    f"but access was attempted from tenant '{tenant_context}'"
                )
            return _row_to_ast_node(row)

    async def get_nodes_by_doc(
        self, doc_id: str, session: AsyncSession | None = None, *, tenant_context: str | None = None
    ) -> list[ASTNode]:
        """Fetch all AST nodes for a given document, scoped to tenant.

        Args:
            doc_id:  The document ID.
            session: Optional existing session.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            A list of :class:`ASTNode` objects.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for get_nodes_by_doc. "
                "All storage operations require a tenant context."
            )
        if session is not None:
            return await self._get_nodes_by_doc(session, doc_id, tenant_context)

        async with self.session() as sess:
            return await self._get_nodes_by_doc(sess, doc_id, tenant_context)

    async def _get_nodes_by_doc(
        self, session: AsyncSession, doc_id: str, tenant_context: str | None = None
    ) -> list[ASTNode]:
        filters = [ASTNodeRow.doc_id == doc_id]
        if tenant_context:
            filters.append(ASTNodeRow.tenant_id == tenant_context)
        stmt = select(ASTNodeRow).where(*filters)
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

    async def delete_node(self, node_id: str, *, tenant_context: str | None = None) -> bool:
        """Delete a node by its ID.

        Args:
            node_id: The UUID4 string identifying the node.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            ``True`` if the node existed and was deleted.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for delete_node. "
                "All storage operations require a tenant context."
            )
        async with self.session() as session:
            row = await session.get(ASTNodeRow, node_id)
            if row is None:
                return False
            # Verify tenant ownership before delete
            if row.tenant_id and row.tenant_id != tenant_context:
                raise PermissionError(
                    f"Node {node_id} belongs to tenant '{row.tenant_id}', "
                    f"cannot delete from tenant '{tenant_context}'"
                )
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

    async def save_temporal_metadata(
        self, meta: TemporalMetadata, *, tenant_context: str | None = None
    ) -> None:
        """Persist temporal metadata for a node using immutable versioning.

        Delegates to :class:`TemporalVersionService.create_version()` to ensure
        historical data is NEVER overwritten.  Each call creates a new immutable
        version row instead of updating the existing one.

        After version creation, the :class:`TemporalMetadataRow` is upserted
        with the caller's metadata values (freshness_score, decay_rate, etc.)
        to ensure they are preserved.

        Args:
            meta: The :class:`TemporalMetadata` to save.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for save_temporal_metadata."
            )
        from apex_rag.temporal.version_service import TemporalVersionService

        version_service = TemporalVersionService(self)
        # Fetch the AST node to get actual content and doc_id for versioning
        ast_node = await self.get_node(meta.node_id, tenant_context=tenant_context)
        node_content = ast_node.content if ast_node else meta.node_id
        node_doc_id = ast_node.doc_id if ast_node else meta.node_id
        await version_service.create_version(
            node_id=meta.node_id,
            content=node_content,
            doc_id=node_doc_id,
            tenant_id=tenant_context,
            source_timestamp=meta.source_date or meta.source_timestamp,
            approval_timestamp=meta.approval_timestamp,
            validity_status=meta.validity_status or "ACTIVE",
        )
        # Upsert the TemporalMetadataRow with the caller's actual metadata values
        async with self.session() as session:
            existing = await session.get(TemporalMetadataRow, meta.node_id)
            now = datetime.now(timezone.utc)
            if existing is not None:
                existing.freshness_score = meta.freshness_score
                existing.decay_rate = meta.decay_rate
                existing.source_date = meta.source_date
                existing.updated_at = now
            else:
                row = TemporalMetadataRow(
                    node_id=meta.node_id,
                    source_date=meta.source_date,
                    ingestion_date=meta.ingestion_date,
                    freshness_score=meta.freshness_score,
                    decay_rate=meta.decay_rate,
                    superseded_by=meta.superseded_by,
                    created_at=now,
                    updated_at=now,
                    effective_from=meta.effective_from or now,
                    effective_to=meta.effective_to,
                    version_number=meta.version_number,
                    is_current=meta.is_current,
                    validity_status=meta.validity_status or "ACTIVE",
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
                created_at=row.created_at,
                updated_at=row.updated_at,
                effective_from=row.effective_from,
                effective_to=row.effective_to,
                version_number=row.version_number,
                revision_number=row.revision_number,
                source_timestamp=row.source_timestamp,
                approval_timestamp=row.approval_timestamp,
                is_current=row.is_current,
                previous_version=row.previous_version,
                validity_status=row.validity_status,
            )

    # ── Causal Edge CRUD ───────────────────────────────────────────────────

    async def _detect_cycle_in_causal_graph(
        self,
        source_node_id: str,
        target_node_id: str,
        session: AsyncSession,
        max_depth: int = 50,
    ) -> bool:
        """Check if adding an edge from ``source_node_id`` to ``target_node_id``
        would create a cycle in the causal graph.

        Uses BFS from ``target_node_id`` to see if it can reach ``source_node_id``
        via existing edges.  If yes, the new edge would create a cycle.

        PRINCIPLE 3 — DAG Lineage.
        Cycle detection occurs during write, never during reads.

        Args:
            source_node_id: The origin node of the proposed edge.
            target_node_id: The destination node of the proposed edge.
            session:        The active database session.
            max_depth:      Maximum BFS depth to prevent unbounded traversal.

        Returns:
            ``True`` if adding the edge would create a cycle.
        """
        # BFS from target_node_id following outgoing edges to see if we can reach source_node_id
        visited: set[str] = {target_node_id}
        bfs_queue: deque[str] = deque([target_node_id])
        depth = 0

        while bfs_queue and depth < max_depth:
            current_id = bfs_queue.popleft()
            if current_id == source_node_id:
                return True
            stmt = select(CausalEdgeRow).where(
                CausalEdgeRow.source_node_id == current_id,
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                if row.target_node_id not in visited:
                    visited.add(row.target_node_id)
                    bfs_queue.append(row.target_node_id)
            depth += 1

        return False

    async def save_causal_edge(self, edge: CausalEdge) -> None:
        """Persist a causal edge with DAG cycle detection.

        PRINCIPLE 3 — DAG Lineage.
        PRINCIPLE 11 — Enforce DAG Acyclicity At Write Time.

        Before inserting a new edge, this method checks whether the
        proposed edge would create a cycle in the causal graph.
        If a cycle is detected, a :class:`ValueError` is raised and
        the edge is **not** persisted.

        Args:
            edge: The :class:`CausalEdge` to save.

        Raises:
            ValueError: If the edge would create a cycle in the causal graph.
        """
        async with self.session() as session:
            edge_type_str = (
                edge.edge_type if isinstance(edge.edge_type, str) else edge.edge_type.value
            )

            # Check for existing edge with same ID — upsert is safe
            existing = await session.get(CausalEdgeRow, edge.edge_id)
            if existing is not None:
                existing.source_node_id = edge.source_node_id
                existing.target_node_id = edge.target_node_id
                existing.edge_type = edge_type_str
                existing.strength = edge.strength
                existing.evidence = edge.evidence
                existing.discovered_at = edge.discovered_at
                return

            # DAG cycle detection: reject if the new edge would create a cycle
            if await self._detect_cycle_in_causal_graph(
                edge.source_node_id,
                edge.target_node_id,
                session,
            ):
                raise ValueError(
                    f"Cannot add causal edge {edge.edge_id}: "
                    f"{edge.source_node_id} → {edge.target_node_id} would create a cycle "
                    f"in the causal graph.  Cycles are rejected at write time "
                    f"(Principle 11 — DAG Acyclicity)."
                )

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

    async def get_edges_for_node(self, node_id: str) -> list[CausalEdge]:
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

    async def save_page_index_entries(self, entries: list[dict[str, Any]]) -> None:
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

    async def get_page_index_entries(self, doc_id: str) -> list[dict[str, Any]]:
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
        self, doc_id: str, query: str, *, tenant_context: str | None = None
    ) -> list[dict[str, Any]]:
        """Full-text search over page index terms (case-insensitive).

        Uses FTS5 when available (SQLite), falls back to ILIKE.

        Args:
            doc_id:  The document ID.
            query:   Search string.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            Matching PageIndexEntry dicts.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError("tenant_context is required for search_page_index.")
        try:
            # Try FTS5 first for full-text on page index
            from apex_rag.retrieval.search.fts5 import FTS5Search

            fts = FTS5Search(self)
            fts_results = await fts.search(query, limit=50)
            if fts_results:
                node_ids = [r["node_id"] for r in fts_results]
                async with self.session() as session:
                    stmt = select(PageIndexEntryRow).where(
                        PageIndexEntryRow.node_id.in_(node_ids),
                        PageIndexEntryRow.doc_id == doc_id,
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
        except Exception:
            pass

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

    async def get_cached_query(self, query_hash: str, doc_id: str) -> dict[str, Any] | None:
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

    async def list_document_ids(self, *, tenant_context: str | None = None) -> list[str]:
        """List all document IDs that have stored AST nodes, scoped to tenant.

        Args:
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            List of document IDs.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for list_document_ids. "
                "All storage operations require a tenant context."
            )
        async with self.session() as session:
            stmt = (
                select(ASTNodeRow.doc_id).distinct().where(ASTNodeRow.tenant_id == tenant_context)
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.all()]

    async def get_document_root_nodes(self, doc_id: str) -> list[ASTNode]:
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
        tenant_context: str | None = None,
    ) -> list[ASTNode]:
        """Search all nodes across all documents using content similarity.

        Uses FTS5 full-text search when available (SQLite), falling back
        to ILIKE.  Results are scoped to the provided tenant context.

        Args:
            query: Search string.
            limit: Maximum results.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            Matching ASTNode objects.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for search_nodes_global. "
                "All storage operations require a tenant context."
            )
        try:
            # Try FTS5 first
            from apex_rag.retrieval.search.fts5 import FTS5Search

            fts = FTS5Search(self)
            fts_results = await fts.search(query, limit=limit)
            if fts_results:
                # Fetch full ASTNode objects from FTS results
                node_ids = [r["node_id"] for r in fts_results]
                async with self.session() as session:
                    stmt = select(ASTNodeRow).where(
                        ASTNodeRow.node_id.in_(node_ids),
                        ASTNodeRow.tenant_id == tenant_context,
                    )
                    result = await session.execute(stmt)
                    rows = result.scalars().all()
                    return [_row_to_ast_node(r) for r in rows]
        except Exception:
            pass

        # Fallback to ILIKE
        async with self.session() as session:
            stmt = (
                select(ASTNodeRow)
                .where(
                    ASTNodeRow.content.ilike(f"%{query}%"),
                    ASTNodeRow.tenant_id == tenant_context,
                )
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_row_to_ast_node(r) for r in rows]

    async def get_document_stats(
        self, doc_id: str, *, tenant_context: str | None = None
    ) -> dict[str, Any]:
        """Return aggregate statistics for a document.

        Args:
            doc_id:  The document ID.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            Dict with keys: doc_id, total_nodes, max_depth, leaf_count.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for get_document_stats. "
                "All storage operations require a tenant context."
            )
        async with self.session() as session:
            stmt = select(ASTNodeRow).where(
                ASTNodeRow.doc_id == doc_id,
                ASTNodeRow.tenant_id == tenant_context,
            )
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

    async def delete_document(self, doc_id: str, *, tenant_context: str | None = None) -> int:
        """Delete all nodes and page index entries for a document.

        Args:
            doc_id:  The document ID.
            tenant_context: Required tenant ID for multi-tenant isolation.

        Returns:
            Number of nodes deleted.

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError

            raise MissingTenantContextError(
                "tenant_context is required for delete_document. "
                "All storage operations require a tenant context."
            )
        async with self.session() as session:
            # Delete page index entries first (scoped to tenant)
            await session.execute(
                sa_text("DELETE FROM apex_page_index WHERE doc_id = :did"),
                {"did": doc_id},
            )
            # Delete causal edges for nodes in this doc
            await session.execute(
                sa_text(
                    "DELETE FROM apex_causal_edges WHERE source_node_id IN "
                    "(SELECT node_id FROM apex_ast_nodes WHERE doc_id = :did AND tenant_id = :tid)"
                ),
                {"did": doc_id, "tid": tenant_context},
            )
            # Delete temporal metadata
            await session.execute(
                sa_text(
                    "DELETE FROM apex_temporal_metadata WHERE node_id IN "
                    "(SELECT node_id FROM apex_ast_nodes WHERE doc_id = :did AND tenant_id = :tid)"
                ),
                {"did": doc_id, "tid": tenant_context},
            )
            # Delete nodes (scoped to tenant)
            result = await session.execute(
                delete(ASTNodeRow).where(
                    ASTNodeRow.doc_id == doc_id,
                    ASTNodeRow.tenant_id == tenant_context,
                )
            )
            return result.rowcount

    # ── Temporal Intelligence CRUD ──────────────────────────────────────────

    async def save_node_version(
        self, version_row: NodeVersionRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a node version row."""

        async def _save(sess: AsyncSession):
            sess.add(version_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_node_versions(
        self, node_id: str, session: AsyncSession | None = None
    ) -> list[NodeVersionRow]:
        """Fetch all versions of a node, ordered by version_number."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(NodeVersionRow)
                .where(NodeVersionRow.node_id == node_id)
                .order_by(NodeVersionRow.version_number)
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def get_node_version_as_of(
        self, node_id: str, as_of: datetime, session: AsyncSession | None = None
    ) -> NodeVersionRow | None:
        """Fetch the node version active/effective as of a specific datetime."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(NodeVersionRow)
                .where(
                    NodeVersionRow.node_id == node_id,
                    NodeVersionRow.effective_from <= as_of,
                    (NodeVersionRow.effective_to.is_(None) | (NodeVersionRow.effective_to > as_of)),
                )
                .order_by(NodeVersionRow.version_number.desc())
            )
            res = await sess.execute(stmt)
            return res.scalars().first()

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def get_nodes_as_of(
        self, doc_id: str, as_of: datetime, session: AsyncSession | None = None
    ) -> list[NodeVersionRow]:
        """Fetch all active node versions for a document as of a specific datetime."""

        async def _get(sess: AsyncSession):
            stmt = select(NodeVersionRow).where(
                NodeVersionRow.doc_id == doc_id,
                NodeVersionRow.effective_from <= as_of,
                (NodeVersionRow.effective_to.is_(None) | (NodeVersionRow.effective_to > as_of)),
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def save_temporal_node(
        self, temporal_node: TemporalNodeRow, session: AsyncSession | None = None
    ) -> None:
        """Save a temporal node row (INSERT-only).

        PRINCIPLE 1 — Immutable Temporal Facts.
        Every change creates a new row instead of mutating existing data.
        The caller is responsible for versioning via :class:`TemporalVersionService`.

        Args:
            temporal_node: The :class:`TemporalNodeRow` to insert.
            session:       Optional existing session.
        """

        async def _save(sess: AsyncSession):
            # INSERT-only — never mutate existing rows
            sess.add(temporal_node)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_temporal_node(
        self, node_id: str, session: AsyncSession | None = None
    ) -> TemporalNodeRow | None:
        """Fetch temporal node context by node ID."""

        async def _get(sess: AsyncSession):
            return await sess.get(TemporalNodeRow, node_id)

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Audit Trail CRUD ───────────────────────────────────────────────────

    async def save_audit_log(
        self, audit_row: AuditLogRow, session: AsyncSession | None = None
    ) -> None:
        """Persist an audit log record."""

        async def _save(sess: AsyncSession):
            sess.add(audit_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_audit_logs(
        self, tenant_id: str | None = None, session: AsyncSession | None = None
    ) -> list[AuditLogRow]:
        """Fetch audit logs, optionally filtered by tenant."""

        async def _get(sess: AsyncSession):
            if tenant_id:
                stmt = (
                    select(AuditLogRow)
                    .where(AuditLogRow.tenant_id == tenant_id)
                    .order_by(AuditLogRow.timestamp.desc())
                )
            else:
                stmt = select(AuditLogRow).order_by(AuditLogRow.timestamp.desc())
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Change History CRUD ────────────────────────────────────────────────

    async def save_change_history(
        self, change_row: ChangeHistoryRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a change history record."""

        async def _save(sess: AsyncSession):
            sess.add(change_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_change_history(
        self, entity_id: str, session: AsyncSession | None = None
    ) -> list[ChangeHistoryRow]:
        """Fetch change history for an entity, ordered by change time."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(ChangeHistoryRow)
                .where(ChangeHistoryRow.entity_id == entity_id)
                .order_by(ChangeHistoryRow.changed_at.desc())
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Timeline Events CRUD ───────────────────────────────────────────────

    async def save_timeline_event(
        self, event_row: TimelineEventRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a timeline event."""

        async def _save(sess: AsyncSession):
            sess.add(event_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_timeline_events(
        self, entity_id: str, session: AsyncSession | None = None
    ) -> list[TimelineEventRow]:
        """Fetch timeline events for an entity, ordered by event date."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(TimelineEventRow)
                .where(TimelineEventRow.entity_id == entity_id)
                .order_by(TimelineEventRow.event_date.asc())
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Role/Field Permissions CRUD ────────────────────────────────────────

    async def save_role_permission(
        self, perm: RolePermissionRow, session: AsyncSession | None = None
    ) -> None:
        """Persist role permission rules."""

        async def _save(sess: AsyncSession):
            sess.add(perm)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_role_permission(
        self, role: str, resource_type: str, action: str, session: AsyncSession | None = None
    ) -> bool:
        """Query if a role is allowed to perform action on a resource type."""

        async def _get(sess: AsyncSession):
            stmt = select(RolePermissionRow.is_allowed).where(
                RolePermissionRow.role == role,
                RolePermissionRow.resource_type == resource_type,
                RolePermissionRow.action == action,
            )
            res = await sess.execute(stmt)
            val = res.scalar()
            return val if val is not None else False

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def save_field_permission(
        self, perm: FieldPermissionRow, session: AsyncSession | None = None
    ) -> None:
        """Persist field permission rules."""

        async def _save(sess: AsyncSession):
            sess.add(perm)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_field_permission(
        self, role: str, resource_type: str, field_name: str, session: AsyncSession | None = None
    ) -> bool:
        """Query if a role is allowed to view/access a field of a resource type."""

        async def _get(sess: AsyncSession):
            stmt = select(FieldPermissionRow.is_allowed).where(
                FieldPermissionRow.role == role,
                FieldPermissionRow.resource_type == resource_type,
                FieldPermissionRow.field_name == field_name,
            )
            res = await sess.execute(stmt)
            val = res.scalar()
            return val if val is not None else False

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Custom Rules CRUD ──────────────────────────────────────────────────

    async def save_custom_rule(
        self, rule: CustomRuleRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a custom security rule."""

        async def _save(sess: AsyncSession):
            sess.add(rule)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_custom_rule(
        self, name: str, session: AsyncSession | None = None
    ) -> CustomRuleRow | None:
        """Fetch a custom security rule by its name."""

        async def _get(sess: AsyncSession):
            stmt = select(CustomRuleRow).where(CustomRuleRow.name == name)
            res = await sess.execute(stmt)
            return res.scalars().first()

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def delete_custom_rule(self, name: str, session: AsyncSession | None = None) -> None:
        """Delete a custom security rule by its name."""

        async def _delete(sess: AsyncSession):
            stmt = delete(CustomRuleRow).where(CustomRuleRow.name == name)
            await sess.execute(stmt)

        if session is not None:
            await _delete(session)
        else:
            async with self.session() as sess:
                await _delete(sess)

    async def save_rule_assignment(
        self, assignment: RuleAssignmentRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a custom rule assignment."""

        async def _save(sess: AsyncSession):
            sess.add(assignment)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_rule_assignments(
        self,
        role: str | None = None,
        user_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> Sequence[RuleAssignmentRow]:
        """Fetch custom rule assignments matching role and/or user_id."""

        async def _get(sess: AsyncSession):
            stmt = select(RuleAssignmentRow)
            filters = []
            if role is not None:
                filters.append(RuleAssignmentRow.role == role)
            if user_id is not None:
                filters.append(RuleAssignmentRow.user_id == user_id)
            if filters:
                stmt = stmt.where(or_(*filters))
            res = await sess.execute(stmt)
            return res.scalars().all()

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def delete_rule_assignment(
        self, assignment_id: int, session: AsyncSession | None = None
    ) -> None:
        """Delete a custom rule assignment by id."""

        async def _delete(sess: AsyncSession):
            stmt = delete(RuleAssignmentRow).where(RuleAssignmentRow.id == assignment_id)
            await sess.execute(stmt)

        if session is not None:
            await _delete(session)
        else:
            async with self.session() as sess:
                await _delete(sess)

    # ── State Snapshots CRUD ───────────────────────────────────────────────

    async def save_state_snapshot(
        self, snapshot_row: StateSnapshotRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a state snapshot."""

        async def _save(sess: AsyncSession):
            sess.add(snapshot_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_state_snapshot(
        self, doc_id: str, as_of: datetime, session: AsyncSession | None = None
    ) -> StateSnapshotRow | None:
        """Fetch state snapshot for document as of a specific date."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(StateSnapshotRow)
                .where(StateSnapshotRow.doc_id == doc_id, StateSnapshotRow.snapshot_date <= as_of)
                .order_by(StateSnapshotRow.snapshot_date.desc())
            )
            res = await sess.execute(stmt)
            return res.scalars().first()

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    # ── Version Lineage CRUD ──────────────────────────────────────

    async def _detect_cycle_in_version_lineage(
        self,
        source_version_id: str,
        target_version_id: str,
        session: AsyncSession,
        max_depth: int = 50,
    ) -> bool:
        """Check if adding a lineage edge from ``source_version_id`` to
        ``target_version_id`` would create a cycle.

        Uses BFS from ``target_version_id`` following outgoing lineage edges
        to see if it can reach ``source_version_id``.

        PRINCIPLE 3 — DAG Lineage.

        Args:
            source_version_id: The origin version of the proposed lineage edge.
            target_version_id: The destination version.
            session:           The active database session.
            max_depth:         Maximum BFS depth.

        Returns:
            ``True`` if adding the edge would create a cycle.
        """
        visited: set[str] = {target_version_id}
        bfs_queue: deque[str] = deque([target_version_id])
        depth = 0

        while bfs_queue and depth < max_depth:
            current_id = bfs_queue.popleft()
            if current_id == source_version_id:
                return True
            stmt = select(VersionLineageRow).where(
                VersionLineageRow.source_version_id == current_id,
            )
            result = await session.execute(stmt)
            for row in result.scalars().all():
                nid = row.target_version_id
                if nid and nid not in visited:
                    visited.add(nid)
                    bfs_queue.append(nid)
            depth += 1

        return False

    async def save_version_lineage(
        self, lineage_row: VersionLineageRow, session: AsyncSession | None = None
    ) -> None:
        """Persist a version lineage entry with DAG cycle detection.

        PRINCIPLE 3 — DAG Lineage.
        PRINCIPLE 11 — Enforce DAG Acyclicity At Write Time.

        Before inserting a new lineage entry, this method checks whether
        the proposed edge would create a cycle.  If so, a :class:`ValueError`
        is raised.

        Args:
            lineage_row: The :class:`VersionLineageRow` to save.

        Raises:
            ValueError: If the edge would create a cycle.
        """

        async def _save(sess: AsyncSession):
            # DAG cycle detection
            source_vid = lineage_row.source_version_id
            target_vid = lineage_row.target_version_id
            if target_vid and await self._detect_cycle_in_version_lineage(
                source_vid,
                target_vid,
                sess,
            ):
                raise ValueError(
                    f"Cannot add version lineage {lineage_row.lineage_id}: "
                    f"{source_vid} → {target_vid} would create a cycle.  "
                    f"Cycles are rejected at write time (Principle 11)."
                )
            sess.add(lineage_row)

        if session is not None:
            await _save(session)
        else:
            async with self.session() as sess:
                await _save(sess)

    async def get_version_lineage(
        self, node_id: str, session: AsyncSession | None = None
    ) -> list[VersionLineageRow]:
        """Fetch version lineage entries for a node, ordered by creation time."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(VersionLineageRow)
                .where(VersionLineageRow.node_id == node_id)
                .order_by(VersionLineageRow.created_at.asc())
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)

    async def resolve_version_lineage_chain(
        self, node_id: str, session: AsyncSession | None = None
    ) -> list[VersionLineageRow]:
        """Traverse the full SUPERSEDES/REPLACED_BY chain to the latest version."""

        async def _traverse(sess: AsyncSession):
            stmt = (
                select(VersionLineageRow)
                .where(VersionLineageRow.node_id == node_id)
                .order_by(VersionLineageRow.created_at.desc())
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _traverse(session)
        async with self.session() as sess:
            return await _traverse(sess)

    async def get_version_lineage_by_type(
        self, node_id: str, lineage_type: str, session: AsyncSession | None = None
    ) -> list[VersionLineageRow]:
        """Fetch version lineage entries filtered by type (e.g. SUPERSEDES, VERSION_OF)."""

        async def _get(sess: AsyncSession):
            stmt = (
                select(VersionLineageRow)
                .where(
                    VersionLineageRow.node_id == node_id,
                    VersionLineageRow.lineage_type == lineage_type,
                )
                .order_by(VersionLineageRow.created_at.asc())
            )
            res = await sess.execute(stmt)
            return list(res.scalars().all())

        if session is not None:
            return await _get(session)
        async with self.session() as sess:
            return await _get(sess)


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
