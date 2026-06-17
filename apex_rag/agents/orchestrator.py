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
            sub_queries = await self.planner.plan(enriched_query)
            logger.info(
                "[ORCHESTRATOR] Plan → %d sub-queries: %s",
                len(sub_queries),
                sub_queries,
            )

            # 2. Navigate
            packets, resolved_sqs = await self._navigate_all(sub_queries, doc_id)

            # Save best effort so far
            if packets:
                best_packets = packets

            if not packets:
                logger.info("[ORCHESTRATOR] No evidence retrieved — aborting.")
                return None

            # 3. Critic
            passes = await self.critic.evaluate(sub_queries, [p.node for p in packets])

            if passes:
                logger.info(
                    "[ORCHESTRATOR] Critic approved after %d iteration(s).",
                    iteration,
                )
                return packets

            # Determine what was missing for the next iteration
            missing_sub_queries = self._find_missing(sub_queries, resolved_sqs)
            missing_context = "; ".join(missing_sub_queries) if missing_sub_queries else "unknown"
            logger.info(
                "[ORCHESTRATOR] Critic rejected — missing: %s",
                missing_context,
            )

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

        # 1. Iterative retrieval (Plan → Navigate → Critic)
        packets = await self.execute_query(query, doc_id, max_iterations=max_iterations)
        if not packets:
            return None

        # 2. Temporal scoring & enrichment
        scorer = self.scorer if self.scorer is not None else FreshnessScorer(domain=domain)
        for pkt in packets:
            # Recompute freshness score if needed
            pkt.temporal_metadata.freshness_score = scorer.compute(pkt.node.source_date)

        mean_freshness = sum(p.temporal_metadata.freshness_score for p in packets) / len(packets)

        # 3. Contradiction detection
        contradictions: list[CausalEdge] = []
        if self.contradiction_detector is not None and len(packets) >= 2:
            contradictions = await self.contradiction_detector.detect_all([p.node for p in packets])

        # 4. Causal graph building
        causal_chain: list[CausalEdge] = []
        if self.causal_builder is not None and len(packets) >= 2:
            graph_edges = await self.causal_builder.build_all(
                [p.node for p in packets],
                include_temporal=True,
                include_semantic=True,
                include_structural=False,
                include_llm=False,
            )
            for ge in graph_edges:
                ce = ge.to_causal_edge()
                if ce is not None:
                    causal_chain.append(ce)

        # 5. Evidence chain building
        if self.causal_retriever is not None and len(packets) >= 2:
            evidence_chain = await self.causal_retriever.build_chain(
                [p.node for p in packets], max_depth=3, max_edges=20
            )
            causal_chain.extend(evidence_chain)

        # 6. Conformal prediction
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

        # 7. Synthesis
        if self.synthesizer is not None:
            answer_text = await self.synthesizer.synthesize(query, conformal_packets)
        else:
            answer_text = self._fallback_synthesis(query, conformal_packets)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

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
