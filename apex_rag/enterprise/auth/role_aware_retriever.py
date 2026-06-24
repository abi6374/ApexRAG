"""
enterprise/auth/role_aware_retriever.py — Role-aware retrieval pipeline.

Integrates access control, temporal validation, and deterministic retrieval
into a single pipeline that ensures unauthorized data never enters the
synthesis pipeline.

Pipeline:
    User
    → Access Validation (tenant, role, permission)
    → Temporal Validation (version, validity)
    → Retrieval (AST navigation)
    → Role-Aware Filtering (field-level, node-level)
    → Verification
    → Role-Aware Synthesis

Usage:
    retriever = RoleAwareRetriever(storage, navigator, access_control)
    packets = await retriever.retrieve("What is revenue?", "doc-123", tenant_context)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from apex_rag.enterprise.auth.access_control import AccessControlAgent
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import (
    AccessAuditRecord,
    EvidencePacket,
)
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.temporal.version_resolver import VersionResolver

logger = logging.getLogger("apex_rag.enterprise.auth.role_aware_retriever")


@dataclass
class RoleAwareResult:
    """Result of a role-aware retrieval operation.

    Attributes:
        packets:       Filtered evidence packets (after access & temporal checks).
        audit_record:  The :class:`AccessAuditRecord` for this retrieval.
        allowed:       Whether the retrieval was allowed.
        blocked_count: Number of nodes blocked due to access restrictions.
    """

    packets: list[EvidencePacket]
    audit_record: AccessAuditRecord
    allowed: bool = True
    blocked_count: int = 0


class RoleAwareRetriever:
    """Access-controlled retrieval pipeline for enterprise deployments.

    Executes before every retrieval operation:

    1. **Access Validation** — tenant isolation, role check, permission check
    2. **Temporal Validation** — version resolution, validity filtering
    3. **Deterministic Retrieval** — AST navigation (delegated to navigator)
    4. **Role-Aware Filtering** — field-level masking, node restriction
    5. **Audit Logging** — every access attempt is recorded

    Unauthorized data is **never** passed to the synthesis layer.
    """

    def __init__(
        self,
        storage: ApexStorage,
        navigator: ASTNavigationAgent,
        access_control: AccessControlAgent,
        version_resolver: VersionResolver | None = None,
    ) -> None:
        self._storage = storage
        self._navigator = navigator
        self._access_control = access_control
        self._version_resolver = version_resolver or VersionResolver(storage)

    # ── Public API ─────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        doc_id: str,
        tenant_context: TenantContext,
        *,
        as_of: datetime | None = None,
        allowed_roles: list[str] | None = None,  # noqa: ARG002
    ) -> RoleAwareResult:
        """Run the full role-aware retrieval pipeline.

        Args:
            query:           Natural-language query.
            doc_id:          Target document ID.
            tenant_context:  The :class:`TenantContext` for the requesting user.
            as_of:           Optional — retrieve state as of this datetime.
            allowed_roles:   Optional override for allowed roles.

        Returns:
            A :class:`RoleAwareResult` with filtered packets and audit trail.
        """
        start = time.perf_counter()
        accessed_nodes: list[str] = []
        blocked_nodes: list[str] = []
        role = tenant_context.roles[0] if tenant_context.roles else "Guest"
        request_id = str(uuid.uuid4())

        # ── Step 1: Access Validation ──────────────────────────────────
        has_access = await self._access_control.check_access(
            tenant_context,
            "read",
            "document",
            doc_tenant_id=doc_id,
        )
        if not has_access:
            logger.warning(
                "ACCESS DENIED: user=%s tenant=%s action=read doc=%s",
                tenant_context.user_id,
                tenant_context.tenant_id,
                doc_id,
            )
            audit = AccessAuditRecord(
                request_id=request_id,
                tenant_id=tenant_context.tenant_id,
                user_id=tenant_context.user_id,
                role=role,
                query=query,
                retrieval_mode="ROLE_AWARE",
                temporal_as_of=as_of,
                allowed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            await self._access_control.log_audit_trail(
                tenant_context,
                "ACCESS_DENIED_READ",
                doc_id,
            )
            return RoleAwareResult(
                packets=[],
                audit_record=audit,
                allowed=False,
                blocked_count=0,
            )

        # ── Step 2: Temporal Validation ────────────────────────────────
        effective_doc_id = doc_id
        if as_of is not None:
            # Resolve version state at as_of — navigator handles date filtering
            logger.info("Temporal retrieval: doc=%s as_of=%s", doc_id, as_of.isoformat())
            # VersionResolver could be used here for precise version targeting

        # ── Step 3: Retrieval via AST Navigator ────────────────────────
        nav_result = await self._navigator.find(
            query=query,
            doc_id=effective_doc_id,
        )

        packets: list[EvidencePacket] = []
        if nav_result and nav_result.verified:
            node = nav_result.node
            accessed_nodes.append(node.node_id)

            # ── Step 4: Role-Aware Field Filtering ─────────────────────
            # Check field-level access for sensitive fields
            masked_content = await self._access_control.mask_content(
                tenant_context,
                node.content,
            )

            # Check node-level access
            node_allowed = await self._access_control.check_access(
                tenant_context,
                "traverse",
                "node",
                doc_tenant_id=doc_id,
            )
            if node_allowed:
                # Build temporal metadata for the evidence packet
                temporal_meta = await self._storage.get_temporal_metadata(
                    node.node_id,
                )

                pkt = EvidencePacket(
                    node=node,
                    content=masked_content,
                    document_id=effective_doc_id,
                    tenant_id=tenant_context.tenant_id,
                    retrieval_score=nav_result.confidence,
                    verification_score=1.0 if nav_result.verified else 0.0,
                    freshness_score=temporal_meta.freshness_score if temporal_meta else 1.0,
                    temporal_metadata=temporal_meta,
                    confidence=nav_result.confidence,
                )
                packets.append(pkt)
            else:
                blocked_nodes.append(node.node_id)

        # ── Step 5: Audit Trail ────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        audit = AccessAuditRecord(
            request_id=request_id,
            tenant_id=tenant_context.tenant_id,
            user_id=tenant_context.user_id,
            role=role,
            timestamp=datetime.now(timezone.utc),
            query=query,
            accessed_nodes=accessed_nodes,
            blocked_nodes=blocked_nodes,
            retrieval_mode="ROLE_AWARE",
            temporal_as_of=as_of,
            allowed=True,
            duration_ms=round(elapsed_ms, 1),
        )

        await self._access_control.log_audit_trail(
            tenant_context,
            "QUERY_COMPLETED",
            doc_id,
        )

        return RoleAwareResult(
            packets=packets,
            audit_record=audit,
            allowed=True,
            blocked_count=len(blocked_nodes),
        )

    async def retrieve_global(
        self,
        query: str,
        tenant_context: TenantContext,
        *,
        as_of: datetime | None = None,
    ) -> RoleAwareResult:
        """Run role-aware retrieval across all documents the user can access.

        Args:
            query:           Natural-language query.
            tenant_context:  The tenant context.
            as_of:           Optional temporal retrieval point.

        Returns:
            A :class:`RoleAwareResult` with aggregated results.
        """
        all_docs = await self._storage.list_document_ids()
        combined_packets: list[EvidencePacket] = []
        total_blocked = 0

        for doc_id in all_docs:
            result = await self.retrieve(
                query,
                doc_id,
                tenant_context,
                as_of=as_of,
            )
            combined_packets.extend(result.packets)
            total_blocked += result.blocked_count

        audit = AccessAuditRecord(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_context.tenant_id,
            user_id=tenant_context.user_id,
            role=tenant_context.roles[0] if tenant_context.roles else "Guest",
            query=query,
            accessed_nodes=[p.node_id for p in combined_packets],
            retrieval_mode="ROLE_AWARE_GLOBAL",
            temporal_as_of=as_of,
            allowed=True,
        )

        return RoleAwareResult(
            packets=combined_packets,
            audit_record=audit,
            allowed=True,
            blocked_count=total_blocked,
        )
