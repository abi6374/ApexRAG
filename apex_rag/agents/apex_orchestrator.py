"""
agents/apex_orchestrator.py — ApexOrchestrator with streaming mode.

Extends the base :class:`Orchestrator` with:

    - A public streaming mode that yields synthesis tokens as they arrive.
    - Integrated audit agents (TemporalAuditAgent, ConformalWrapperAgent).
    - A convenience method ``run()`` that returns a fully populated
      :class:`ApexAnswer` with provenance.

Usage::

    orchestrator = ApexOrchestrator(
        planner=planner,
        navigator=navigator,
        critic=critic,
        synthesizer=synthesizer,
        temporal_auditor=temporal_audit_agent,
        conformal_wrapper=conformal_wrapper_agent,
    )

    # Streaming
    async for token in orchestrator.stream("What is Q3 revenue?", "doc1"):
        print(token, end="", flush=True)

    # Non-streaming
    answer = await orchestrator.run("What is Q3 revenue?", "doc1")
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from apex_rag.agents.audit.conformal_wrapper import ConformalWrapperAgent
from apex_rag.agents.audit.temporal_audit import TemporalAuditAgent
from apex_rag.agents.orchestrator import Orchestrator
from apex_rag.enterprise.auth.access_control import AccessControlAgent
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.models.unified_models import ApexAnswer
from apex_rag.observability.telemetry import get_tracer

logger = logging.getLogger("apex_rag.agents.apex_orchestrator")

# OpenTelemetry tracer for enterprise tracing
_otel_tracer = get_tracer("apex_rag.agents.apex_orchestrator")


class ApexOrchestrator(Orchestrator):
    """High-level orchestrator with streaming and full provenance.

    Extends :class:`Orchestrator` with:

    - **Streaming mode**: ``stream()`` yields tokens as the synthesizer
      generates them, with citations resolved via evidence packets.
    - **Temporal audit**: Automatically runs :class:`TemporalAuditAgent`
      to detect contradictions and stale evidence.
    - **Conformal wrapper**: Applies conformal prediction to produce the
      ``coverage_guarantee`` and ``prediction_set_size`` fields.

    Args:
        temporal_auditor:   Optional :class:`TemporalAuditAgent` instance.
                            Creates a default one if not provided.
        conformal_wrapper:  Optional :class:`ConformalWrapperAgent` instance.
                            Creates a default one if not provided.
        **kwargs:           All other arguments are passed to the base
                            :class:`Orchestrator`.
    """

    def __init__(
        self,
        *,
        temporal_auditor: TemporalAuditAgent | None = None,
        conformal_wrapper: ConformalWrapperAgent | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._temporal_auditor = temporal_auditor or TemporalAuditAgent()
        self._conformal_wrapper = conformal_wrapper or ConformalWrapperAgent(
            coverage_level=0.90,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def temporal_auditor(self) -> TemporalAuditAgent:
        """The temporal audit agent used by this orchestrator."""
        return self._temporal_auditor

    @property
    def conformal_wrapper(self) -> ConformalWrapperAgent:
        """The conformal wrapper agent used by this orchestrator."""
        return self._conformal_wrapper

    # ── Run (non-streaming) ──────────────────────────────────────────

    async def run(
        self,
        query: str,
        doc_id: str,
        *,
        max_iterations: int | None = None,
        domain: str = "general",  # noqa: ARG002
        calibration_scores: list[float] | None = None,
        ablation_mode: bool = False,
        tenant_context: TenantContext | None = None,
    ) -> ApexAnswer | None:
        """Execute the full pipeline and return a populated ApexAnswer.

        This is the primary non-streaming entry point.  It:

        1. Runs the Plan → Navigate → Critic loop.
        2. Scores temporal freshness and detects contradictions (skipped in ablation).
        3. Applies conformal prediction (skipped in ablation).
        4. Synthesises the final answer.
        5. Returns an :class:`ApexAnswer` with all provenance.

        Args:
            query:              The user's natural-language query.
            doc_id:             Target document ID.
            max_iterations:     Maximum refinement iterations (default: 3).
            domain:             Domain for freshness decay (default "general").
            calibration_scores: Optional conformal calibration scores.
            ablation_mode:      If True, run Layer 1 (AST) only, skipping
                                temporal, causal, and conformal layers.

        Returns:
            An :class:`ApexAnswer`, or ``None`` if no evidence found.
        """
        start = time.perf_counter()

        # Check for temporal query using TemporalReasoningAgent
        import json
        import uuid

        from apex_rag.temporal.state_reconstructor import StateReconstructor
        from apex_rag.temporal.temporal_agent import TemporalReasoningAgent
        from apex_rag.temporal.temporal_retriever import TemporalRetriever

        retriever = TemporalRetriever(self.navigator._storage)
        reconstructor = StateReconstructor(self.navigator._storage)
        temporal_agent = TemporalReasoningAgent(retriever, reconstructor)

        has_temporal_history = False
        if not ablation_mode and temporal_agent.detect_time_query(query):
            storage = self.navigator._storage
            if storage is not None:
                try:
                    latest_nodes = await retriever.get_latest_nodes(doc_id)
                    if latest_nodes:
                        has_temporal_history = True
                    elif hasattr(storage, "get_timeline_events"):
                        res = storage.get_timeline_events(doc_id)
                        if inspect.isawaitable(res):
                            events = await res
                        else:
                            events = res
                        if events:
                            has_temporal_history = True
                except Exception:
                    pass

        if has_temporal_history:
            temp_res = await temporal_agent.solve_temporal_query(query, doc_id)

            evidence_packets = []
            answer_text = temp_res.get("reasoning", "")
            res_data = temp_res.get("result", {})
            content_str = json.dumps(res_data) if res_data else ""
            if "content" in res_data:
                content_str = res_data["content"]
            elif "summary" in res_data:
                content_str = res_data["summary"]

            from apex_rag.models.unified_models import ASTNode as UnifiedASTNode
            from apex_rag.models.unified_models import EvidencePacket as UnifiedEvidencePacket
            from apex_rag.models.unified_models import NodeType, TemporalMetadata

            node = UnifiedASTNode(
                node_id=str(uuid.uuid4()),
                content=content_str or "Temporal reasoning results",
                node_type=NodeType.PARAGRAPH,
                doc_id=doc_id,
            )
            pkt = UnifiedEvidencePacket(
                node=node,
                temporal_metadata=TemporalMetadata(node_id=node.node_id, freshness_score=1.0),
                retrieval_score=1.0,
                content=node.content,
            )
            evidence_packets.append(pkt)

            if tenant_context is not None:
                ac_agent = AccessControlAgent(self.navigator._storage)
                for ep in evidence_packets:
                    ep.content = await ac_agent.mask_content(tenant_context, ep.content)
                answer_text = await ac_agent.mask_content(tenant_context, answer_text)
                await ac_agent.log_audit_trail(tenant_context, "TEMPORAL_QUERY", doc_id)

            elapsed = (time.perf_counter() - start) * 1000

            return ApexAnswer(
                answer_text=answer_text,
                evidence_packets=evidence_packets,
                temporal_freshness=1.0,
                contradictions=[],
                coverage_guarantee=1.0,
                prediction_set_size=len(evidence_packets),
                causal_chain=[],
                query=query,
                latency_ms=round(elapsed, 1),
            )

        # 1. Iterative retrieval
        unified_packets = await self.execute_query(
            query, doc_id, max_iterations=max_iterations, tenant_context=tenant_context
        )

        if not unified_packets:
            return None

        from apex_rag.models.unified_models import (
            EvidencePacket as UnifiedEvidencePacket,
        )

        # Default values for metadata
        contradictions = []
        mean_freshness = 1.0
        coverage_guarantee = 1.0
        prediction_set_size = len(unified_packets)
        filtered_packets = unified_packets

        if not ablation_mode:
            # 3. Temporal audit
            audit_result = await self._temporal_auditor.audit(
                unified_packets,
                doc_id=doc_id,
            )
            contradictions = audit_result.conflicts
            mean_freshness = audit_result.mean_freshness

            # 4. Conformal prediction
            conformal_result = self._conformal_wrapper.wrap(
                unified_packets,
                calibration_scores=calibration_scores,
            )
            filtered_packets = conformal_result.filtered_packets
            coverage_guarantee = conformal_result.coverage_guarantee
            prediction_set_size = conformal_result.prediction_set_size

        # 5. Synthesis
        if self.synthesizer is not None:
            core_packets = self._unified_to_core(filtered_packets)
            answer_text = await self.synthesizer.synthesize(query, core_packets)
        else:
            answer_text = self._fallback_synthesis(
                query,
                self._unified_to_core(filtered_packets),
            )

        elapsed = (time.perf_counter() - start) * 1000

        if tenant_context is not None:
            ac_agent = AccessControlAgent(self.navigator._storage)
            answer_text = await ac_agent.mask_content(tenant_context, answer_text)

        return ApexAnswer(
            answer_text=answer_text,
            evidence_packets=filtered_packets,
            temporal_freshness=round(mean_freshness, 4),
            contradictions=contradictions,
            coverage_guarantee=coverage_guarantee,
            prediction_set_size=prediction_set_size,
            causal_chain=list(contradictions),
            query=query,
            latency_ms=round(elapsed, 1),
        )

    # ── Streaming ────────────────────────────────────────────────────

    async def stream(
        self,
        query: str,
        doc_id: str,
        *,
        max_iterations: int | None = None,
        domain: str = "general",  # noqa: ARG002
        tenant_context: TenantContext | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the answer tokens as they are generated.

        Internally runs the full retrieval pipeline, then streams the
        synthesis output token-by-token.  Citations in the output use
        the ``[Node ID: <id>]`` format.

        Args:
            query:          The user's natural-language query.
            doc_id:         Target document ID.
            max_iterations: Maximum refinement iterations (default: 3).
            domain:         Domain for freshness decay (default "general").

        Yields:
            Answer token chunks as they arrive from the LLM.
        """
        # 1. Iterative retrieval (traced)
        with _otel_tracer.start_as_current_span("apex_orchestrator.stream.retrieve") as span:
            span.set_attribute("query", query)
            span.set_attribute("doc_id", doc_id)
            span.set_attribute("max_iterations", max_iterations or self.max_iterations)

            base_packets = await self.execute_query(
                query, doc_id, max_iterations=max_iterations, tenant_context=tenant_context
            )

            span.set_attribute("packets_retrieved", len(base_packets) if base_packets else 0)

        if not base_packets:
            yield "I could not find enough evidence to answer your query."
            return

        # 2. Stream synthesis (traced)
        with _otel_tracer.start_as_current_span("apex_orchestrator.stream.synthesize") as span:
            span.set_attribute("query", query)
            span.set_attribute("packets_count", len(base_packets))
            token_count = 0
            if self.synthesizer is not None:
                async for chunk in self.synthesizer.stream_synthesize(
                    query,
                    base_packets,
                ):
                    token_count += 1
                    yield chunk
            else:
                text = self._fallback_synthesis(query, base_packets)
                token_count = 1
                yield text
            span.set_attribute("token_count", token_count)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _unified_to_core(
        packets: list[Any],
    ) -> list[Any]:
        """Convert unified EvidencePackets to core EvidencePackets for synthesis."""
        from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket

        core: list[CoreEvidencePacket] = []
        for pkt in packets:
            node_id = pkt.node.node_id if hasattr(pkt, "node") and pkt.node else "unknown"
            content = pkt.node.content if hasattr(pkt, "node") and pkt.node else ""
            core.append(
                CoreEvidencePacket(
                    node_id=node_id,
                    source_document="unknown",
                    section_path=content[:60] if content else "",
                    page_number=None,
                    paragraph_index=None,
                    bounding_box=None,
                    retrieval_reason="ApexOrchestrator retrieval",
                    verification_result=True,
                    confidence_score=getattr(pkt, "retrieval_score", 0.5),
                    content=content,
                )
            )
        return core
