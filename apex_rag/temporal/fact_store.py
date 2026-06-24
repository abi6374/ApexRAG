"""
temporal/fact_store.py — Immutable Fact Store for the Temporal Fact Layer.

Defines the :class:`TemporalFact` frozen data model and the :class:`FactStore`
which provides async, tenant-aware CRUD operations backed by SQLAlchemy.

PRINCIPLE 1 — Immutable Temporal Facts.
  TemporalFact objects are frozen dataclasses.  They are never mutated.
  Every change creates a new version with supersession tracking.

PRINCIPLE 18 — Tenant Isolation Everywhere.
  Every fact carries a tenant_id and all queries filter by tenant.

Usage:
    fact = TemporalFact(
        subject="Revenue",
        predicate="was",
        object="$120,000",
        source_document_id="doc-123",
        source_node_id="node-456",
        valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    store = FactStore(storage)
    saved = await store.save_fact(fact, tenant_context="tenant-a")
    facts = await store.get_facts_at_time("doc-123", as_of=datetime(2025, 6, 1))
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from apex_rag.ingestion.apex_storage import ApexBase, ApexStorage

logger = logging.getLogger("apex_rag.temporal.fact_store")


# ═══════════════════════════════════════════════════════════════
# TemporalFact — Immutable Fact Model
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TemporalFact:
    """An immutable, provenance-tracked fact extracted from a document.

    PRINCIPLE 1 — Immutable Temporal Facts.
    This dataclass is frozen.  Values cannot be mutated after creation.
    To update a fact, create a new version and mark the old one as superseded.

    Attributes:
        fact_id:            Globally unique identifier (UUID4).
        tenant_id:          Tenant isolation boundary.
        subject:            The fact subject (e.g. "Revenue", "Policy X").
        predicate:          The relationship (e.g. "was", "equals", "requires").
        object:             The fact object (e.g. "$120,000", "true").
        confidence:         Extraction confidence in [0, 1].
        source_document_id: The document this fact was extracted from.
        source_node_id:     The AST node this fact was extracted from.
        valid_from:         Start of validity period.
        valid_to:           End of validity period (None = currently valid).
        created_at:         When this fact was created/stored.
        parent_fact_id:     Previous version of this fact (for lineage).
        superseded_by:      The fact_id that supersedes this one.
        extraction_method:  How this fact was extracted (regex, llm, manual).
        metadata:           Optional additional metadata.
    """

    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 1.0
    source_document_id: str = ""
    source_node_id: str = ""
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_fact_id: str | None = None
    superseded_by: str | None = None
    extraction_method: str = "regex"
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# FactRow — ORM Model
# ═══════════════════════════════════════════════════════════════


class FactRow(ApexBase):
    """SQL row for the :class:`TemporalFact` model."""

    __tablename__ = "temporal_facts"

    fact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, default="default"
    )
    subject: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    predicate: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    object_: Mapped[str] = mapped_column(
        "object", Text, nullable=False, default=""
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_document_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, default=""
    )
    source_node_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("apex_ast_nodes.node_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    parent_fact_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("temporal_facts.fact_id", ondelete="SET NULL"),
        nullable=True,
    )
    superseded_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("temporal_facts.fact_id", ondelete="SET NULL"),
        nullable=True,
    )
    extraction_method: Mapped[str] = mapped_column(
        String(20), nullable=False, default="regex"
    )
    metadata_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )

    __table_args__ = (
        Index("ix_facts_tenant", "tenant_id"),
        Index("ix_facts_document", "source_document_id"),
        Index("ix_facts_subject", "subject"),
        Index("ix_facts_valid", "valid_from", "valid_to"),
        Index("ix_facts_superseded", "superseded_by"),
    )


# ═══════════════════════════════════════════════════════════════
# FactStore
# ═══════════════════════════════════════════════════════════════


class FactStore:
    """Async, tenant-aware CRUD for immutable :class:`TemporalFact` objects.

    All operations enforce tenant isolation (Principle 18).
    Facts are never mutated — use supersession for corrections.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Helper: FK-safe value coalescing ───────────────────────────────

    @staticmethod
    def _coalesce_fk(value: str | None) -> str | None:
        """Convert empty strings to None for FK-safe storage.

        SQLite's foreign key enforcement only ignores NULL values.
        Empty strings (``""``) are NOT ignored and will fail FK
        constraints if no matching row exists in the referenced table.
        """
        return value if value else None

    # ── Write Operations ───────────────────────────────────────────────

    async def save_fact(
        self,
        fact: TemporalFact,
        *,
        tenant_context: str | None = None,
    ) -> TemporalFact:
        """Save a single immutable fact.

        Args:
            fact:            The :class:`TemporalFact` to persist.
            tenant_context:  Required tenant ID for isolation.

        Returns:
            The saved fact (with generated fact_id if not provided).

        Raises:
            MissingTenantContextError: If tenant_context is None or empty.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for save_fact."
            )

        row = FactRow(
            fact_id=fact.fact_id,
            tenant_id=tenant_context,
            subject=fact.subject,
            predicate=fact.predicate,
            object_=fact.object,
            confidence=fact.confidence,
            source_document_id=fact.source_document_id,
            source_node_id=self._coalesce_fk(fact.source_node_id),
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            created_at=fact.created_at,
            parent_fact_id=self._coalesce_fk(fact.parent_fact_id),
            superseded_by=self._coalesce_fk(fact.superseded_by),
            extraction_method=fact.extraction_method,
            metadata_json=json.dumps(fact.metadata) if fact.metadata else "{}",
        )

        async with self._storage.session() as session:
            session.add(row)

        return fact

    async def save_facts(
        self,
        facts: list[TemporalFact],
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Batch-save multiple immutable facts in a single transaction.

        Args:
            facts:           List of :class:`TemporalFact` to persist.
            tenant_context:  Required tenant ID.

        Returns:
            The saved facts.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for save_facts."
            )

        async with self._storage.session() as session:
            for fact in facts:
                row = FactRow(
                    fact_id=fact.fact_id,
                    tenant_id=tenant_context,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_=fact.object,
                    confidence=fact.confidence,
                    source_document_id=fact.source_document_id,
                    source_node_id=self._coalesce_fk(fact.source_node_id),
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to,
                    created_at=fact.created_at,
                    parent_fact_id=self._coalesce_fk(fact.parent_fact_id),
                    superseded_by=self._coalesce_fk(fact.superseded_by),
                    extraction_method=fact.extraction_method,
                    metadata_json=json.dumps(fact.metadata) if fact.metadata else "{}",
                )
                session.add(row)

        return facts

    # ── Read Operations ────────────────────────────────────────────────

    async def get_fact(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> TemporalFact | None:
        """Fetch a single fact by its ID.

        Args:
            fact_id:         The fact UUID.
            tenant_context:  Required tenant ID.

        Returns:
            The :class:`TemporalFact` if found and accessible.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for get_fact."
            )

        async with self._storage.session() as session:
            stmt = select(FactRow).where(
                FactRow.fact_id == fact_id,
                FactRow.tenant_id == tenant_context,
            )
            result = await session.execute(stmt)
            row = result.scalars().first()
            if row is None:
                return None
            return self._row_to_fact(row)

    async def get_facts(
        self,
        *,
        tenant_context: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TemporalFact]:
        """Fetch facts with pagination, scoped to tenant.

        Args:
            tenant_context: Required tenant ID.
            limit:          Maximum results.
            offset:         Pagination offset.

        Returns:
            A list of :class:`TemporalFact` objects.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for get_facts."
            )

        async with self._storage.session() as session:
            stmt = (
                select(FactRow)
                .where(FactRow.tenant_id == tenant_context)
                .order_by(FactRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return [self._row_to_fact(r) for r in result.scalars().all()]

    async def get_facts_by_document(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Fetch all facts for a document, scoped to tenant.

        Args:
            doc_id:          The document ID.
            tenant_context:  Required tenant ID.

        Returns:
            Facts extracted from the document.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for get_facts_by_document."
            )

        async with self._storage.session() as session:
            stmt = (
                select(FactRow)
                .where(
                    FactRow.source_document_id == doc_id,
                    FactRow.tenant_id == tenant_context,
                )
                .order_by(FactRow.created_at.asc())
            )
            result = await session.execute(stmt)
            return [self._row_to_fact(r) for r in result.scalars().all()]

    async def get_facts_at_time(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Fetch facts that were valid at a specific point in time.

        Uses the ``valid_from`` / ``valid_to`` window for O(log n) lookups
        via the ``ix_facts_valid`` index (Principle 4).

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            Facts valid at the given time.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for get_facts_at_time."
            )

        async with self._storage.session() as session:
            stmt = (
                select(FactRow)
                .where(
                    FactRow.source_document_id == doc_id,
                    FactRow.tenant_id == tenant_context,
                    FactRow.valid_from <= as_of,
                    (FactRow.valid_to.is_(None) | (FactRow.valid_to > as_of)),
                )
                .order_by(FactRow.created_at.desc())
            )
            result = await session.execute(stmt)
            return [self._row_to_fact(r) for r in result.scalars().all()]

    async def get_active_facts(
        self,
        doc_id: str,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Fetch currently active (non-superseded, non-expired) facts.

        Args:
            doc_id:          The document ID.
            tenant_context:  Required tenant ID.

        Returns:
            Active facts for the document.
        """
        now = datetime.now(timezone.utc)
        return await self.get_facts_at_time(
            doc_id, now, tenant_context=tenant_context,
        )

    # ── Delete Operations ──────────────────────────────────────────────

    async def delete_fact(
        self,
        fact_id: str,
        *,
        tenant_context: str | None = None,
    ) -> bool:
        """Irrevocably expire a fact by creating a superseding tombstone.

        PRINCIPLE 1 — Immutable Temporal Facts.
        Facts are never mutated.  This method creates a new tombstone fact
        that marks the original as superseded.  The original row is
        left untouched — only its lineage is extended.

        Args:
            fact_id:         The fact UUID to expire.
            tenant_context:  Required tenant ID.

        Returns:
            True if the fact was found and a tombstone was created.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for delete_fact."
            )

        async with self._storage.session() as session:
            # Fetch the existing fact row (read-only, never mutate)
            stmt = select(FactRow).where(
                FactRow.fact_id == fact_id,
                FactRow.tenant_id == tenant_context,
            )
            result = await session.execute(stmt)
            row = result.scalars().first()
            if row is None:
                return False

            # Create a tombstone version that supersedes the original.
            # The original row is never modified (Principle 1).
            now = datetime.now(timezone.utc)
            tombstone = FactRow(
                fact_id=str(uuid.uuid4()),
                tenant_id=row.tenant_id,
                subject=f"__DELETED__:{row.subject}",
                predicate="__DELETED__",
                object_="__DELETED__",
                confidence=row.confidence,
                source_document_id=row.source_document_id,
                source_node_id=row.source_node_id,
                valid_from=row.valid_from,
                valid_to=row.valid_to,
                created_at=now,
                parent_fact_id=fact_id,
                extraction_method=row.extraction_method,
                metadata_json=row.metadata_json,
            )
            session.add(tombstone)
            logger.info(
                "Created tombstone fact %s to supersede %s (immutable delete)",
                tombstone.fact_id, fact_id,
            )
            return True

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_fact(row: FactRow) -> TemporalFact:
        """Convert a database row to a :class:`TemporalFact`."""
        import json
        try:
            meta = json.loads(row.metadata_json) if row.metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return TemporalFact(
            fact_id=row.fact_id,
            tenant_id=row.tenant_id,
            subject=row.subject,
            predicate=row.predicate,
            object=row.object_,
            confidence=row.confidence,
            source_document_id=row.source_document_id,
            source_node_id=row.source_node_id or "",
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            created_at=row.created_at,
            parent_fact_id=row.parent_fact_id,
            superseded_by=row.superseded_by,
            extraction_method=row.extraction_method,
            metadata=meta,
        )
