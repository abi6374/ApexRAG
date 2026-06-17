"""
agents/orchestrator.py — Enhanced multi-agent orchestrator with iterative refinement.

Integrates all four layers:
    - Part 1-2: AST navigation (Planner → Navigator → Critic)
    - Part 3: Temporal intelligence (FreshnessScorer, ContradictionDetector)
    - Part 4: Causal knowledge graph (CausalGraphBuilder, CausalRetriever)
    - Part 5: Iterative refinement, full synthesis pipeline

The orchestrator runs a Plan → Navigate → Critic loop with up to N iterations.
If the Critic rejects, it re-plans with context about what's missing.
After success, it enriches the result with temporal metadata, causal edges,
and evidence chains before passing everything to the synthesizer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from apex_rag.agents.synthesizer.agent import EvidenceSynthesizerAgent
from apex_rag.core.ast.models import ASTNode as CoreASTNode
from apex_rag.core.evidence.models import EvidencePacket as CoreEvidencePacket
from apex_rag.core.protocols.interfaces import CriticAgent, QueryPlanner
from apex_rag.graph.edges.causal_builder import CausalGraphBuilder
from apex_rag.graph.edges.causal_retriever import CausalRetriever
from apex_rag.models.unified_models import (
    ApexAnswer,
    ASTNode as UnifiedASTNode,
    CausalEdge,
    EdgeType,
    EvidencePacket as UnifiedEvidencePacket,
    NodeType,
    TemporalMetadata,
)
from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
from apex_rag.retrieval.conformal.predictor import ConformalPredictor
from apex_rag.retrieval.conformal.scorer import (
    NonconformityScorer,
    NonconformityStrategy,
)
from apex_rag.temporal.contradiction import TemporalContradictionDetector
from apex_rag.temporal.scorer import FreshnessScorer
from apex_rag.retrieval.state.machine import RetrievalStateMachine, RetrievalState
from apex_rag.observability.trace_manager import trace_manager
from apex_rag.graph.reasoning_engine import GraphReasoningEngine
from apex_rag.temporal.lineage import DocumentLineageEngine
from apex_rag.temporal.resolver import ContradictionResolver
from apex_rag.utils import ReasoningTrace, logger

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_MAX_ITERATIONS = 3


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


class Orchestrator:
    """
    Coordinates the Planner, Navigator, and Critic for multi-hop graph reasoning.

    Supports **iterative refinement**: if the Critic rejects the retrieved
    context, the Planner re-decomposes the original query with added context
    about what was missing.  This loop runs up to ``max_iterations`` times.

    The integrated flow (``execute_query_integrated``) additionally wires in:
        - **Temporal scoring** (freshness decay per node)
        - **Temporal contradictions** (CONTRADICTS edges between dated nodes)
        - **Causal graph** (structural + semantic + temporal edges)
        - **Evidence chains** (reasoning paths via CausalRetriever)
        - **Final synthesis** (LLM answer with inline citations)
    """

    def __init__(
        self,
        planner: QueryPlanner,
        navigator: ASTNavigationAgent,
        critic: CriticAgent,
        *,
        synthesizer: EvidenceSynthesizerAgent | None = None,
        causal_builder: CausalGraphBuilder | None = None,
        causal_retriever: CausalRetriever | None = None,
        scorer: FreshnessScorer | None = None,
        contradiction_detector: TemporalContradictionDetector | None = None,
        conformal_predictor: ConformalPredictor | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        trace: ReasoningTrace | None = None,
    ) -> None:
        self.planner = planner
        self.navigator = navigator
        self.critic = critic
        self.synthesizer = synthesizer
        self.causal_builder = causal_builder
        self.causal_retriever = causal_retriever
        self.scorer = scorer  # None → fallback created in execute_query_integrated with domain
        self.contradiction_detector = contradiction_detector
        self.conformal_predictor = conformal_predictor
        self.max_iterations = max_iterations
        self.trace = trace or ReasoningTrace(enabled=True)

    # ── Basic execution (Plan → Navigate → Critic) ────────────────────

    async def execute_query(
        self,
        query: str,
        doc_id: str,
        *,
        max_iterations: int | None = None,
    ) -> list[UnifiedEvidencePacket] | None:
        """
        Execute the reasoning loop with iterative refinement.

        If the Critic rejects, the planner is re-invoked with context
        about *which* sub-queries were not answered, and the loop retries.
        Runs up to ``max_iterations`` times.

        Args:
            query:          The user's natural-language query.
            doc_id:         Target document ID.
            max_iterations: Maximum refinement iterations (default: 3).

        Returns:
            A list of verified :class:`UnifiedEvidencePacket` objects, or
            ``None`` if the query could not be answered.
        """
        iters = max_iterations if max_iterations is not None else self.max_iterations
        missing_context: str = ""
        best_packets: list[UnifiedEvidencePacket] | None = None

        trace_id = getattr(self, "trace_id", None) or f"trace-{int(time.time() * 1000)}"
        self.trace_id = trace_id
        trace_manager.start_trace(trace_id)
        state_machine = RetrievalStateMachine(query_id=trace_id)

        state_machine.transition_to(RetrievalState.QUERY_CLASSIFIED, {"query": query})
        trace_manager.publish(trace_id, "reasoning", "QUERY_CLASSIFIED", {"query": query})

        for iteration in range(1, iters + 1):
            logger.info(
                "[ORCHESTRATOR] Iteration %d/%d — planning...",
                iteration,
                iters,
            )

            # 1. Plan (with missing context if applicable)
            enriched_query = query
            if missing_context:
                enriched_query = (
                    f"{query}\n\n[Previous attempt missing: {missing_context}]"
                )

            # Use plan_query if available to support classification
            query_type = "FACTUAL"
            from apex_rag.agents.planner.agent import QueryPlannerAgent
            if isinstance(self.planner, QueryPlannerAgent) and hasattr(self.planner, "plan_query"):
                planner_data = await self.planner.plan_query(enriched_query)
                query_type = planner_data.get("query_type", "FACTUAL")
                sub_queries = planner_data.get("sub_queries", [query])
            else:
                sub_queries = await self.planner.plan(enriched_query)

            state_machine.transition_to(
                RetrievalState.PLAN_GENERATED, 
                {"sub_queries": sub_queries, "iteration": iteration, "query_type": query_type}
            )
            trace_manager.publish(trace_id, "reasoning", "PLAN_GENERATED", {
                "sub_queries": sub_queries,
                "iteration": iteration,
                "query_type": query_type
            })

            # 2. Navigate
            state_machine.transition_to(RetrievalState.NAVIGATION_RUNNING)
            trace_manager.publish(trace_id, "navigation", "NAVIGATION_RUNNING", {"iteration": iteration})

            packets, resolved_sqs = await self._navigate_all(sub_queries, doc_id)

            # Transition to VERIFICATION_RUNNING
            state_machine.transition_to(RetrievalState.VERIFICATION_RUNNING)
            trace_manager.publish(trace_id, "verification", "VERIFICATION_RUNNING", {
                "iteration": iteration,
                "packets_count": len(packets) if packets else 0
            })

            # Save best effort so far
            if packets:
                best_packets = packets

            if not packets:
                logger.info("[ORCHESTRATOR] No evidence retrieved — aborting.")
                state_machine.transition_to(RetrievalState.FAILED, {"reason": "No evidence retrieved"})
                trace_manager.publish(trace_id, "navigation", "NAVIGATION_FAILED", {})
                return None

            # 3. Critic
            state_machine.transition_to(RetrievalState.CRITIC_REVIEW)
            trace_manager.publish(trace_id, "critic", "CRITIC_REVIEW", {})

            passes = await self.critic.evaluate(sub_queries, [p.node for p in packets if p.node])

            if passes:
                logger.info(
                    "[ORCHESTRATOR] Critic approved after %d iteration(s).",
                    iteration,
                )
                state_machine.transition_to(RetrievalState.COMPLETED, {"iterations": iteration})
                trace_manager.publish(trace_id, "critic", "CRITIC_APPROVED", {"iterations": iteration})
                return packets

            # Determine what was missing for the next iteration
            missing_sub_queries = self._find_missing(sub_queries, resolved_sqs)
            missing_context = "; ".join(missing_sub_queries) if missing_sub_queries else "unknown"
            logger.info(
                "[ORCHESTRATOR] Critic rejected — missing: %s",
                missing_context,
            )

            state_machine.record_retry(RetrievalState.PLAN_GENERATED)
            state_machine.rollback_to(RetrievalState.PLAN_GENERATED, reason=f"Critic rejected, missing: {missing_context}")

        logger.warning(
            "[ORCHESTRATOR] Exceeded max iterations (%d) — returning best effort.",
            iters,
        )
        # Return the best effort from the last successful iteration
        return best_packets

    # ── Integrated execution (temporal + causal + synthesis) ───────────

    async def execute_query_integrated(
        self,
        query: str,
        doc_id: str,
        *,
        max_iterations: int | None = None,
        domain: str = "general",
    ) -> ApexAnswer | None:
        """
        Full integrated pipeline: iterative retrieval → temporal scoring →
        contradiction detection → causal graph → evidence chains → synthesis.

        Returns a fully populated :class:`ApexAnswer` with all provenance.

        Args:
            query:          The user's natural-language query.
            doc_id:         Target document ID.
            max_iterations: Maximum refinement iterations (default: 3).
            domain:         Domain for freshness decay rate (default "general").

        Returns:
            An :class:`ApexAnswer` with answer text, evidence, temporal
            freshness, contradictions, causal chain, and latency, or
            ``None`` if no evidence could be retrieved.
        """
        start_time = time.perf_counter()

        trace_id = getattr(self, "trace_id", None) or f"trace-{int(time.time() * 1000)}"
        self.trace_id = trace_id
        trace_manager.start_trace(trace_id)
        state_machine = RetrievalStateMachine(query_id=trace_id)

        # 1. Iterative retrieval (Plan → Navigate → Critic)
        packets = await self.execute_query(query, doc_id, max_iterations=max_iterations)
        if not packets:
            state_machine.transition_to(RetrievalState.FAILED, {"reason": "No evidence retrieved in integrated run"})
            return None

        # 2. Temporal scoring & enrichment
        scorer = self.scorer if self.scorer is not None else FreshnessScorer(domain=domain)
        for pkt in packets:
            if pkt.node:
                pkt.temporal_metadata.freshness_score = scorer.compute(pkt.node.source_date)
                pkt.freshness_score = pkt.temporal_metadata.freshness_score

        # V3 Graph Reasoning stage
        state_machine.transition_to(RetrievalState.GRAPH_REASONING)
        trace_manager.publish(trace_id, "graph", "GRAPH_REASONING", {})
        
        causal_chain = []
        contradictions = []

        # Graph Reasoning BFS Traversal
        reasoning_engine = GraphReasoningEngine(storage=self.navigator._storage)
        seed_nodes = [p.node for p in packets if p.node]
        if seed_nodes:
            reasoning_chain = await reasoning_engine.build_reasoning_chain(seed_nodes)
            causal_chain.extend(reasoning_chain.edges)
            contradictions.extend(reasoning_chain.contradictions)
            trace_manager.publish(trace_id, "graph", "REASONING_CHAIN_BUILT", {
                "edges_count": len(reasoning_chain.edges),
                "score": reasoning_chain.score,
                "contradictions_count": len(reasoning_chain.contradictions)
            })

        # 3. Temporal audit / Lineage and contradiction resolution
        state_machine.transition_to(RetrievalState.TEMPORAL_AUDIT)
        trace_manager.publish(trace_id, "temporal", "TEMPORAL_AUDIT", {})

        lineage_engine = DocumentLineageEngine()
        lineage_engine.register_version(doc_id, version="1.0")

        if seed_nodes:
            active_nodes = lineage_engine.filter_obsolete_nodes(seed_nodes)
            active_ids = {n.node_id for n in active_nodes}
            packets = [p for p in packets if p.node_id in active_ids or (p.node and p.node.node_id in active_ids)]

        # Run contradiction resolver to handle conflicts
        resolver = ContradictionResolver(lineage_engine=lineage_engine)
        contradiction_report = resolver.resolve(packets, causal_chain + contradictions)
        packets = contradiction_report.authoritative_packets
        
        # Add resolver contradictions if any
        for conflict in contradiction_report.conflicts:
            for edge in causal_chain + contradictions:
                if edge.edge_id == conflict.edge_id and edge not in contradictions:
                    contradictions.append(edge)

        # Call contradiction_detector if configured for mock compatibility
        if self.contradiction_detector is not None and len(packets) >= 2:
            detected_contradictions = await self.contradiction_detector.detect_all([p.node for p in packets if p.node])
            for dc in detected_contradictions:
                if dc not in contradictions:
                    contradictions.append(dc)

        trace_manager.publish(trace_id, "temporal", "LINEAGE_AUDIT_COMPLETE", {
            "has_conflicts": contradiction_report.has_conflicts or (len(contradictions) > 0),
            "conflicts_count": len(contradictions),
            "authoritative_count": len(packets)
        })

        mean_freshness = sum(p.freshness_score for p in packets) / len(packets) if packets else 1.0

        # Run normal builders for backward compatibility or extra coverage
        if self.causal_builder is not None and len(packets) >= 2:
            graph_edges = await self.causal_builder.build_all(
                [p.node for p in packets if p.node],
                include_temporal=True,
                include_semantic=True,
                include_structural=False,
                include_llm=False,
            )
            for ge in graph_edges:
                ce = ge.to_causal_edge()
                if ce is not None and ce not in causal_chain:
                    causal_chain.append(ce)

        # 5. Evidence chain building (legacy)
        if self.causal_retriever is not None and len(packets) >= 2:
            evidence_chain = await self.causal_retriever.build_chain(
                [p.node for p in packets if p.node], max_depth=3, max_edges=20
            )
            for ec in evidence_chain:
                if ec not in causal_chain:
                    causal_chain.append(ec)

        # 6. Conformal prediction
        state_machine.transition_to(RetrievalState.CONFORMAL_FILTERING)
        trace_manager.publish(trace_id, "conformal", "CONFORMAL_FILTERING", {})

        conformal_packets = list(packets)
        coverage_guarantee = 0.0
        prediction_set_size = len(packets)

        if self.conformal_predictor is not None:
            conformal_scores = self.conformal_predictor.scorer.score_many(packets)
            for pkt, score in zip(packets, conformal_scores):
                pkt.nonconformity_score = score

            calibrator = self.conformal_predictor.calibrator
            if calibrator is not None:
                threshold = calibrator.calibrate(conformal_scores) if len(conformal_scores) >= calibrator.min_calibration_size else 0.0
                if threshold > 0.0:
                    filtered, guarantee, set_size = self.conformal_predictor.predict(
                        packets, threshold
                    )
                    conformal_packets = filtered
                    coverage_guarantee = guarantee
                    prediction_set_size = set_size
                else:
                    coverage_guarantee = calibrator.coverage_level
                    
        trace_manager.publish(trace_id, "conformal", "CONFORMAL_COMPLETE", {
            "coverage_guarantee": coverage_guarantee,
            "prediction_set_size": prediction_set_size
        })

        # 7. Synthesis
        state_machine.transition_to(RetrievalState.SYNTHESIS)
        trace_manager.publish(trace_id, "synthesis", "SYNTHESIS", {})

        if self.synthesizer is not None:
            answer_text = await self.synthesizer.synthesize(query, conformal_packets)
        else:
            answer_text = self._fallback_synthesis(query, conformal_packets)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        state_machine.transition_to(RetrievalState.COMPLETED)
        trace_manager.publish(trace_id, "synthesis", "COMPLETED", {"latency_ms": elapsed_ms})

        return ApexAnswer(
            answer_text=answer_text,
            evidence_packets=conformal_packets,
            temporal_freshness=round(mean_freshness, 4),
            contradictions=contradictions,
            coverage_guarantee=coverage_guarantee,
            prediction_set_size=prediction_set_size,
            causal_chain=causal_chain,
            query=query,
            latency_ms=round(elapsed_ms, 1),
        )

    # ── Internal helpers ──────────────────────────────────────────────

    async def _navigate_all(
        self,
        sub_queries: list[str],
        doc_id: str,
    ) -> tuple[list[UnifiedEvidencePacket], set[str]]:
        """Run navigation for each sub-query and collect results.

        Returns:
            A tuple of (packets, resolved_sub_queries).
        """
        packets: list[UnifiedEvidencePacket] = []
        resolved: set[str] = set()

        for i, sq in enumerate(sub_queries):
            logger.info("[NAVIGATE] Resolving sub-query: '%s'", sq)
            nav_result = await self.navigator.find(query=sq, doc_id=doc_id)

            if nav_result and nav_result.verified:
                logger.info(
                    "[NAVIGATE] Node %s answers '%s'", nav_result.node_id, sq
                )
                
                # Build UnifiedEvidencePacket directly
                pkt = UnifiedEvidencePacket(
                    node=nav_result.node,
                    temporal_metadata=TemporalMetadata(
                        node_id=nav_result.node_id,
                        source_date=nav_result.node.source_date,
                        freshness_score=nav_result.confidence,
                    ),
                    causal_edges=[], # To be enriched by builder
                    retrieval_score=nav_result.confidence,
                    nonconformity_score=1.0, # Default
                    rank=len(packets) + 1,
                )
                packets.append(pkt)
                resolved.add(sq)
            else:
                logger.info("[NAVIGATE] Failed to resolve sub-query: '%s'", sq)

        return packets, resolved

    @staticmethod
    def _find_missing(
        sub_queries: list[str],
        resolved: set[str],
    ) -> list[str]:
        """Determine which sub-queries were not answered by the retrieved nodes."""
        if not resolved:
            return sub_queries
        return [sq for sq in sub_queries if sq not in resolved]

    @staticmethod
    def _fallback_synthesis(
        query: str, packets: list[UnifiedEvidencePacket]
    ) -> str:
        """Fallback when no synthesizer is configured."""
        parts = [f"**Answer for:** {query}", ""]
        for i, pkt in enumerate(packets, 1):
            parts.append(f"**[Source {i}]** (Node: {pkt.node.node_id})")
            parts.append(pkt.node.content[:200])
            parts.append("")
        return "\n".join(parts)
