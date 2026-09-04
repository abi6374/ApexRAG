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

    # ── Conformal Calibration ────────────────────────────────────────

    async def calibrate_conformal(
        self,
        calibration_examples: list[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Calibrate the conformal-prediction coverage guarantee from a
        held-out labeled calibration set.

        Without calling this, ``ApexIndex.query()``'s conformal threshold
        stays at its uncalibrated default (0.0), which means every
        retrieved packet passes through unfiltered and
        ``answer.coverage_guarantee`` reports ``0.0`` — the "statistically
        grounded coverage guarantee" is decorative until calibration runs
        at least once. This method makes it real.

        Standard split-conformal calibration (Angelopoulos & Bates, 2022):
        for each labeled example, run the query, find the nonconformity
        score of the evidence packet that actually contains the gold
        answer (the "true" packet), and calibrate the threshold from the
        distribution of those scores. A calibration example whose gold
        answer wasn't retrieved at all is scored as maximally nonconforming
        (1.0) rather than dropped, so retrieval misses still count against
        the guarantee instead of being silently excluded.

        Because the underlying :class:`ConformalWrapperAgent` keeps its
        calibrated threshold as instance state on the orchestrator, calling
        this once makes **every subsequent** ``index.query()`` call use the
        calibrated threshold automatically — no other API changes needed.

        Args:
            calibration_examples: A held-out list of
                ``(question, doc_id, gold_answer_substring)`` tuples —
                held out meaning disjoint from whatever queries you'll
                report results on. Needs at least ``min_calibration_size``
                (default 10) examples with valid scores, or the threshold
                falls back to a conservative 0.0 (matching today's
                behavior) — see :class:`ConformalCalibrator`.

        Returns:
            A summary dict: ``threshold``, ``n_examples``,
            ``n_retrieval_hits`` (how many calibration queries actually
            retrieved their gold answer), ``coverage_level``, and
            ``calibrated`` (``False`` if too few examples produced a
            real threshold).

        Usage::

            summary = await index.enterprise.calibrate_conformal([
                ("What was Q1 revenue?", "doc1", "$10M"),
                ("Who is the CEO?", "doc1", "Jane Smith"),
                # ... at least 10 examples, held out from your eval set
            ])
            print(summary)  # {"threshold": 0.42, "calibrated": True, ...}

            # Now real:
            answer = await index.query("What was Q2 revenue?", "doc1")
            print(answer.coverage_guarantee)  # reflects the calibrated level
        """
        wrapper = self._orchestrator.conformal_wrapper
        scorer = wrapper.scorer

        scores: list[float] = []
        n_hits = 0
        for question, doc_id, gold_answer in calibration_examples:
            answer = await self._orchestrator.run(
                query=question,
                doc_id=doc_id,
                ablation_mode=True,  # raw retrieval packets; conformal/temporal not needed here
            )
            if answer is None or not answer.evidence_packets:
                scores.append(1.0)
                continue

            gold_lower = gold_answer.lower()
            matching = [p for p in answer.evidence_packets if gold_lower in p.content.lower()]
            if matching:
                n_hits += 1
                scores.append(min(scorer.score_many(matching)))  # type: ignore[arg-type]
            else:
                # Gold answer wasn't retrieved -- worst-case nonconformity,
                # not silently dropped.
                scores.append(1.0)

        threshold = wrapper.calibrate(scores)
        return {
            "threshold": threshold,
            "n_examples": len(calibration_examples),
            "n_retrieval_hits": n_hits,
            "coverage_level": wrapper.calibrator.coverage_level,
            "calibrated": threshold > 0.0,
        }
