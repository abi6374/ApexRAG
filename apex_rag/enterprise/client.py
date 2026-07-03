"""
enterprise/client.py — Enterprise extension for ApexRAG.

``EnterpriseClient`` is the dedicated entry point for all enterprise-grade
capabilities: temporal versioning, RBAC queries, and compliance features.

Usage::

    from apex_rag import ApexIndex

    index = await ApexIndex.create()
    enterprise = index.enterprise

    # Temporal query
    result = await enterprise.temporal_query(
        "What was revenue in Q1?", doc_id,
        as_of=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )

    # Role-aware query
    answer = await enterprise.role_aware_query(
        "What is net profit?", doc_id,
        TenantContext(tenant_id="acme", user_id="alice", roles=["Analyst"]),
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from apex_rag.enterprise.auth.access_control import AccessControlAgent
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.auth.role_aware_retriever import RoleAwareRetriever
from apex_rag.enterprise.auth.role_aware_synthesis import RoleAwareSynthesis
from apex_rag.models.unified_models import ApexAnswer
from apex_rag.temporal.reasoning_service import TemporalReasoningService
from apex_rag.temporal.version_resolver import VersionResolver

if TYPE_CHECKING:
    from apex_rag.client import ApexIndex


class EnterpriseClient:
    """Enterprise-grade query and compliance interface for ApexRAG.

    Provides temporal versioning, role-based access control (RBAC),
    audit trails, and data isolation — everything needed for regulated
    environments.

    Constructed automatically via ``ApexIndex.enterprise``.

    Args:
        index: The parent :class:`ApexIndex` instance.
    """

    def __init__(self, index: ApexIndex) -> None:
        self._index = index
        self._storage = index._storage
        self._orchestrator = index._orchestrator
        self._llm = index._llm

        # Lazily constructed service singletons
        self._version_resolver: VersionResolver | None = None
        self._temporal_reasoning: TemporalReasoningService | None = None
        self._access_control: AccessControlAgent | None = None

    # ── Service properties (lazy singletons) ──────────────────────────

    @property
    def version_resolver(self) -> VersionResolver:
        """Access the version resolver for temporal version resolution."""
        if self._version_resolver is None:
            self._version_resolver = VersionResolver(self._storage)
        return self._version_resolver

    @property
    def temporal_reasoning(self) -> TemporalReasoningService:
        """Access the temporal reasoning service."""
        if self._temporal_reasoning is None:
            self._temporal_reasoning = TemporalReasoningService(self._storage)
        return self._temporal_reasoning

    @property
    def access_control(self) -> AccessControlAgent:
        """Access the access control agent for RBAC operations."""
        if self._access_control is None:
            self._access_control = AccessControlAgent(self._storage)
        return self._access_control

    # ── Temporal Query API ──────────────────────────────────────────

    async def temporal_query(
        self,
        question: str,
        doc_id: str,
        *,
        as_of: datetime | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        latest: bool = False,
    ) -> dict[str, Any]:
        """Query the index with temporal awareness.

        Supports the following temporal modes:
          - **Latest**: ``latest=True`` — returns current state
          - **As-of date**: ``as_of=datetime(2025, 1, 15)`` — state at a point in time
          - **Date range**: ``start_date=..., end_date=...`` — state over a range

        Examples::

            result = await index.enterprise.temporal_query(
                "What is revenue?", doc_id,
                as_of=datetime(2025, 1, 15, tzinfo=timezone.utc),
            )

        Args:
            question:   Natural-language question.
            doc_id:     Target document ID.
            as_of:      Retrieve version state as of this datetime.
            start_date: Start of date range for range queries.
            end_date:   End of date range for range queries.
            latest:     If True, return the latest active state.

        Returns:
            Dict with keys: mode, result, reasoning, provenance, latency_ms.
        """
        service = self.temporal_reasoning
        return await service.answer(
            question,
            doc_id,
            as_of=as_of,
            start_date=start_date,
            end_date=end_date,
            latest=latest,
        )

    async def get_version_history(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Get the full version history for a node.

        Args:
            node_id: The node ID.

        Returns:
            List of version snapshots with temporal metadata.
        """
        return await self.temporal_reasoning.get_version_history(node_id)

    async def get_version_lineage(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Get the version lineage chain for a node.

        Traces the SUPERSEDES/REPLACED_BY chain to show how a node
        evolved through versions.

        Args:
            node_id: The node ID.

        Returns:
            List of lineage entries.
        """
        return await self.temporal_reasoning.get_lineage(node_id)

    async def temporal_compare(
        self,
        question: str,
        doc_id: str,
        date_a: datetime,
        date_b: datetime,
    ) -> dict[str, Any]:
        """Compare document/metric state between two points in time.

        Args:
            question: Contextual query describing what to compare.
            doc_id:   Target document ID.
            date_a:   First point in time.
            date_b:   Second point in time.

        Returns:
            Dict with before/after metrics, diffs, and change analysis.
        """
        return await self.temporal_reasoning.compare(
            question,
            doc_id,
            date_a,
            date_b,
        )

    # ── Role-Aware Query API ────────────────────────────────────────

    async def role_aware_query(
        self,
        question: str,
        doc_id: str,
        tenant_context: TenantContext,
        *,
        as_of: datetime | None = None,
    ) -> ApexAnswer:
        """Query with enterprise RBAC enforcement.

        Runs the full role-aware retrieval pipeline:
          1. Access validation (tenant + role + permission)
          2. Temporal validation (version resolution)
          3. AST navigation
          4. Role-aware filtering (field-level masking)
          5. Audit trail logging

        Args:
            question:        Natural-language query.
            doc_id:          Target document ID.
            tenant_context:  The :class:`TenantContext` for the requesting user.
            as_of:           Optional — retrieve state as of this datetime.

        Returns:
            An :class:`ApexAnswer` with only the content the user is
            authorized to see.
        """
        navigator = self._orchestrator.navigator
        rar = RoleAwareRetriever(self._storage, navigator, self.access_control)

        result = await rar.retrieve(
            question,
            doc_id,
            tenant_context,
            as_of=as_of,
        )

        if not result.allowed or not result.packets:
            return ApexAnswer(
                answer_text=("Access denied or no authorized evidence found for your query."),
                query=question,
                evidence_packets=[],
            )

        rasynthesis = RoleAwareSynthesis(
            self._orchestrator.synthesizer,
            self.access_control,
        )

        answer_text = await rasynthesis.synthesize(
            tenant_context,
            question,
            result.packets,
        )

        freshness_scores = [p.freshness_score for p in result.packets]
        mean_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 1.0

        return ApexAnswer(
            answer_text=answer_text,
            evidence_packets=result.packets,
            temporal_freshness=mean_freshness,
            query=question,
            coverage_guarantee=1.0,
            prediction_set_size=len(result.packets),
        )
