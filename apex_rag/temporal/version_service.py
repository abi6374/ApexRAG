"""
temporal/version_service.py — Immutable Temporal Version Service.

Replaces the previous UPDATE-based save_temporal_metadata() with a
fully immutable versioning system.  Every node change creates a new
NodeVersionRow instead of updating existing rows, guaranteeing that
historical data is NEVER overwritten.

Key design:
  - Every mutation creates a new version row (INSERT-only).
  - Old versions have effective_to set and is_current=False.
  - New versions have effective_to=NULL and is_current=True.
  - content_hash enables integrity verification.
  - Historical reconstruction works for any timestamp via the
    effective_from/effective_to window.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from apex_rag.ingestion.apex_storage import (
    ApexStorage,
    NodeVersionRow,
)
from apex_rag.ingestion.apex_storage import (
    TemporalMetadataRow as TMRow,
)
from apex_rag.models.unified_models import TemporalMetadata

logger = logging.getLogger("apex_rag.temporal.version_service")


class TemporalVersionService:
    """Immutable version management for AST nodes.

    All version mutations are INSERT-only.  UPDATE is only used to
    close the effective_to window on the previous version when a new
    version is created.

    Methods:
        create_version()        — Create a new version for a node.
        resolve_version_at_time() — Find the version active at a timestamp.
        get_version_chain()     — Get full version ancestry.
        get_latest_version()    — Get the most current active version.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Public API ─────────────────────────────────────────────────────────

    async def create_version(
        self,
        node_id: str,
        content: str,
        doc_id: str,
        *,
        tenant_id: str = "default",
        source_timestamp: datetime | None = None,
        approval_timestamp: datetime | None = None,
        validity_status: str = "ACTIVE",
        session: Any = None,
    ) -> NodeVersionRow:
        """Create a new immutable version for a node.

        This method:
          1. Closes the previous version (sets effective_to, is_current=False).
          2. Inserts a new NodeVersionRow with incremented version_number.
          3. Persists the new version to the database.

        Args:
            node_id:            The node ID being versioned.
            content:            The new content for this version.
            doc_id:             The document ID.
            tenant_id:          Tenant isolation boundary.
            source_timestamp:   When the source document was authored.
            approval_timestamp: When this version was approved.
            validity_status:    Status (ACTIVE, PENDING, DRAFT, etc.).
            session:            Optional existing async session.

        Returns:
            The newly created :class:`NodeVersionRow`.
        """
        now = datetime.now(timezone.utc)
        content_hash = self._compute_content_hash(content)

        if session is not None:
            return await self._create_version_in_session(
                session,
                node_id,
                content,
                doc_id,
                tenant_id,
                source_timestamp,
                approval_timestamp,
                validity_status,
                content_hash,
                now,
            )

        async with self._storage.session() as sess:
            return await self._create_version_in_session(
                sess,
                node_id,
                content,
                doc_id,
                tenant_id,
                source_timestamp,
                approval_timestamp,
                validity_status,
                content_hash,
                now,
            )

    async def resolve_version_at_time(
        self,
        node_id: str,
        as_of: datetime,
        *,
        session: Any = None,
    ) -> NodeVersionRow | None:
        """Resolve the version active at a specific point in time.

        Uses the effective_from / effective_to validity window.
        If multiple versions overlap, the highest version_number wins.

        Args:
            node_id: The node ID.
            as_of:   The target datetime.
            session: Optional existing async session.

        Returns:
            The :class:`NodeVersionRow` active at ``as_of``, or None.
        """
        if session is not None:
            return await self._resolve_version_at_time_in_session(
                session,
                node_id,
                as_of,
            )

        async with self._storage.session() as sess:
            return await self._resolve_version_at_time_in_session(
                sess,
                node_id,
                as_of,
            )

    async def get_version_chain(
        self,
        node_id: str,
        *,
        session: Any = None,
    ) -> list[NodeVersionRow]:
        """Get the full version ancestry for a node.

        Returns versions ordered by version_number ascending (oldest first).

        Args:
            node_id: The node ID.
            session: Optional existing async session.

        Returns:
            All versions for the node, oldest first.
        """
        if session is not None:
            return await self._get_version_chain_in_session(session, node_id)

        async with self._storage.session() as sess:
            return await self._get_version_chain_in_session(sess, node_id)

    async def get_latest_version(
        self,
        node_id: str,
        *,
        session: Any = None,
    ) -> NodeVersionRow | None:
        """Get the latest current (active) version of a node.

        Args:
            node_id: The node ID.
            session: Optional existing async session.

        Returns:
            The latest :class:`NodeVersionRow` with is_current=True, or None.
        """
        if session is not None:
            return await self._get_latest_version_in_session(session, node_id)

        async with self._storage.session() as sess:
            return await self._get_latest_version_in_session(sess, node_id)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        """Compute a SHA-256 content hash for integrity verification."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _create_version_in_session(
        self,
        session: Any,
        node_id: str,
        content: str,
        doc_id: str,
        tenant_id: str,
        source_timestamp: datetime | None,
        approval_timestamp: datetime | None,  # noqa: ARG002
        validity_status: str,
        content_hash: str,
        now: datetime,
    ) -> NodeVersionRow:
        # 1. Close the previous current version (if any)
        await session.execute(
            update(NodeVersionRow)
            .where(
                NodeVersionRow.node_id == node_id,
                NodeVersionRow.is_current.is_(True),
            )
            .values(
                effective_to=now,
                is_current=False,
            )
        )

        # 2. Determine the next version number
        latest = await self._get_latest_version_in_session(session, node_id)
        next_version_number = (latest.version_number + 1) if latest else 1
        previous_version_id = latest.version_id if latest else None

        # 3. Insert the new version row
        new_version = NodeVersionRow(
            version_id=str(uuid.uuid4()),
            node_id=node_id,
            content=content,
            created_at=now,
            updated_at=now,
            effective_from=now,
            effective_to=None,
            version_number=next_version_number,
            revision_number=0,
            source_timestamp=source_timestamp,
            is_current=True,
            superseded_by=None,
            previous_version=previous_version_id,
            validity_status=validity_status,
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash=content_hash,
        )
        session.add(new_version)

        # 4. Sync TemporalMetadataRow (create if not exists, update otherwise)
        existing_meta = await session.get(TMRow, node_id)
        if existing_meta is not None:
            existing_meta.version_number = next_version_number
            existing_meta.is_current = True
            existing_meta.effective_from = now
            existing_meta.effective_to = None
            existing_meta.updated_at = now
        else:
            # First-time creation — insert a new TemporalMetadataRow
            meta = TemporalMetadata(
                node_id=node_id,
                freshness_score=1.0,
                version_number=next_version_number,
                is_current=True,
                effective_from=now,
                ingestion_date=now,
                source_date=source_timestamp or now,
            )

            trow = TMRow(
                node_id=meta.node_id,
                source_date=meta.source_date,
                ingestion_date=meta.ingestion_date,
                freshness_score=meta.freshness_score,
                decay_rate=meta.decay_rate,
                created_at=now,
                updated_at=now,
                effective_from=meta.effective_from or now,
                effective_to=meta.effective_to,
                version_number=meta.version_number,
                is_current=meta.is_current,
                validity_status=validity_status,
            )
            session.add(trow)

        logger.info(
            "Created version %d for node %s (doc=%s, tenant=%s)",
            next_version_number,
            node_id,
            doc_id,
            tenant_id,
        )
        return new_version

    @staticmethod
    async def verify_content_integrity(version: NodeVersionRow) -> bool:
        """Verify that a version's content hash matches its actual content.

        Recomputes the SHA-256 hash of the content and compares it to the
        stored ``content_hash``.  This detects data corruption or tampering.

        Args:
            version: The :class:`NodeVersionRow` to verify.

        Returns:
            ``True`` if the content matches the stored hash.
        """
        expected_hash = TemporalVersionService._compute_content_hash(version.content)
        return version.content_hash == expected_hash

    async def _resolve_version_at_time_in_session(
        self,
        session: Any,
        node_id: str,
        as_of: datetime,
    ) -> NodeVersionRow | None:
        stmt = (
            select(NodeVersionRow)
            .where(
                NodeVersionRow.node_id == node_id,
                NodeVersionRow.effective_from <= as_of,
                (NodeVersionRow.effective_to.is_(None) | (NodeVersionRow.effective_to > as_of)),
            )
            .order_by(NodeVersionRow.version_number.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalars().first()

    async def _get_version_chain_in_session(
        self,
        session: Any,
        node_id: str,
    ) -> list[NodeVersionRow]:
        stmt = (
            select(NodeVersionRow)
            .where(NodeVersionRow.node_id == node_id)
            .order_by(NodeVersionRow.version_number.asc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def _get_latest_version_in_session(
        self,
        session: Any,
        node_id: str,
    ) -> NodeVersionRow | None:
        # NOTE: Must query by node_id only (not is_current=True) because the
        # previous version's is_current was already set to False by the caller
        # before this method is called.
        stmt = (
            select(NodeVersionRow)
            .where(NodeVersionRow.node_id == node_id)
            .order_by(NodeVersionRow.version_number.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalars().first()
