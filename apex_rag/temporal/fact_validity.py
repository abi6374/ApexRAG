"""
temporal/fact_validity.py — Fact Validity Resolver.

PRINCIPLE 4 — O(log n) Validity Resolution.
  Historical lookups use indexed queries on ``valid_from`` / ``valid_to``
  via the ``ix_facts_valid`` index.  No full-table scans.

PRINCIPLE 5 — Lazy Snapshot Construction.
  State is resolved on demand when ``resolve_at_time()`` is called.
  No pre-built snapshots are created during ingestion.

Resolves which facts are valid at specific points in time, building on
the low-level :class:`FactStore` CRUD to provide high-level temporal
resolution methods.

Usage:
    resolver = FactValidityResolver(fact_store)
    facts = await resolver.resolve_at_time("doc-123", as_of=datetime(2025, 6, 1), tenant_context="tenant-a")
    latest = await resolver.resolve_latest("doc-123", subject="Revenue", tenant_context="tenant-a")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apex_rag.temporal.fact_store import FactStore, TemporalFact

logger = logging.getLogger("apex_rag.temporal.fact_validity")


class FactValidityResolver:
    """High-level temporal fact resolution engine.

    Provides convenience methods for resolving which facts are valid
    at specific points in time.  All methods delegate to
    :class:`FactStore` for indexed queries.

    Principles:
      - **4**: All lookups use indexed queries on ``valid_from`` / ``valid_to``
      - **5**: State is resolved lazily on demand — no pre-built snapshots
      - **18**: All methods require ``tenant_context`` for tenant isolation

    Attributes:
        fact_store: The :class:`FactStore` instance to delegate to.
    """

    def __init__(self, fact_store: FactStore) -> None:
        self._fact_store = fact_store

    @property
    def fact_store(self) -> FactStore:
        """Access the underlying FactStore."""
        return self._fact_store

    # ── Primary Resolution Methods ────────────────────────────────────────

    async def resolve_at_time(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve all facts valid at a specific point in time.

        PRINCIPLE 4 — O(log n) lookup using the ``ix_facts_valid`` index.

        A fact is considered valid at ``as_of`` if:
          - ``valid_from <= as_of`` AND
          - (``valid_to IS NULL`` OR ``valid_to > as_of``)

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime for validity resolution.
            tenant_context:  Required tenant ID for isolation.

        Returns:
            Facts valid at the given time.
        """
        return await self._fact_store.get_facts_at_time(
            doc_id, as_of, tenant_context=tenant_context,
        )

    async def resolve_latest(
        self,
        doc_id: str,
        subject: str | None = None,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve the latest (most recently created) facts.

        PRINCIPLE 4 — Indexed lookup on ``ix_facts_document``.

        Args:
            doc_id:          The document ID.
            subject:         Optional — filter by subject to narrow results.
            tenant_context:  Required tenant ID.

        Returns:
            The most recent facts, sorted by ``created_at`` descending.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for resolve_latest."
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )
        if subject:
            facts = [f for f in facts if f.subject == subject]

        facts.sort(key=lambda f: f.created_at, reverse=True)
        return facts

    async def resolve_between(
        self,
        doc_id: str,
        start: datetime,
        end: datetime,
        *,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve facts that were valid at any point in a date range.

        PRINCIPLE 4 — Uses indexed window queries.

        A fact is considered valid in the range ``[start, end]`` if its
        validity window overlaps with the query range:
          - ``valid_from <= end AND (valid_to IS NULL OR valid_to > start)``

        Args:
            doc_id:          The document ID.
            start:           Start of the query range.
            end:             End of the query range.
            tenant_context:  Required tenant ID.

        Returns:
            Facts valid within the range.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for resolve_between."
            )

        # Fetch all facts for the document, then filter by window overlap
        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )
        result = [
            f
            for f in facts
            if f.valid_from <= end
            and (f.valid_to is None or f.valid_to > start)
        ]
        result.sort(key=lambda f: f.valid_from)
        return result

    async def resolve_before(
        self,
        doc_id: str,
        before: datetime,
        *,
        subject: str | None = None,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve facts that were valid entirely before a cutoff time.

        PRINCIPLE 4 — Uses indexed validity window queries.

        A fact is considered valid before ``before`` if:
          - ``valid_to IS NOT NULL AND valid_to <= before``

        In other words, facts that had already expired by the cutoff.

        Args:
            doc_id:          The document ID.
            before:          The cutoff datetime.
            subject:         Optional — filter by subject.
            tenant_context:  Required tenant ID.

        Returns:
            Facts that expired on or before the cutoff.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for resolve_before."
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )
        result = [
            f
            for f in facts
            if f.valid_to is not None and f.valid_to <= before
        ]
        if subject:
            result = [f for f in result if f.subject == subject]
        result.sort(key=lambda f: f.valid_to)
        return result

    async def resolve_after(
        self,
        doc_id: str,
        after: datetime,
        *,
        subject: str | None = None,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve facts that became valid after a cutoff time.

        PRINCIPLE 4 — Uses indexed validity window queries.

        A fact is considered valid after ``after`` if:
          - ``valid_from >= after``

        Args:
            doc_id:          The document ID.
            after:           The cutoff datetime.
            subject:         Optional — filter by subject.
            tenant_context:  Required tenant ID.

        Returns:
            Facts that became valid on or after the cutoff.
        """
        if not tenant_context:
            from apex_rag.enterprise.auth.access_control import MissingTenantContextError
            raise MissingTenantContextError(
                "tenant_context is required for resolve_after."
            )

        facts = await self._fact_store.get_facts_by_document(
            doc_id, tenant_context=tenant_context,
        )
        result = [f for f in facts if f.valid_from >= after]
        if subject:
            result = [f for f in result if f.subject == subject]
        result.sort(key=lambda f: f.valid_from)
        return result

    # ── Convenience ───────────────────────────────────────────────────────

    async def resolve_current(
        self,
        doc_id: str,
        *,
        subject: str | None = None,
        tenant_context: str | None = None,
    ) -> list[TemporalFact]:
        """Resolve facts that are currently valid (at this moment).

        Shorthand for ``resolve_at_time(doc_id, datetime.now(timezone.utc))``.

        Args:
            doc_id:          The document ID.
            subject:         Optional — filter by subject.
            tenant_context:  Required tenant ID.

        Returns:
            Currently valid facts.
        """
        now = datetime.now(timezone.utc)
        facts = await self.resolve_at_time(
            doc_id, now, tenant_context=tenant_context,
        )
        if subject:
            facts = [f for f in facts if f.subject == subject]
        return facts

    async def resolve_snapshot(
        self,
        doc_id: str,
        as_of: datetime,
        *,
        tenant_context: str | None = None,
    ) -> dict[str, list[TemporalFact]]:
        """Resolve a full snapshot of facts at a point in time, grouped by subject.

        PRINCIPLE 5 — Lazy Snapshot Construction.
        The snapshot is built on demand (never pre-built during ingestion).

        Args:
            doc_id:          The document ID.
            as_of:           The target datetime.
            tenant_context:  Required tenant ID.

        Returns:
            Dict mapping ``subject`` → list of facts valid at the given time.
        """
        facts = await self.resolve_at_time(
            doc_id, as_of, tenant_context=tenant_context,
        )
        snapshot: dict[str, list[TemporalFact]] = {}
        for fact in facts:
            snapshot.setdefault(fact.subject, []).append(fact)
        return snapshot
