"""
graph/dags/reasoning_dag.py — ReasoningDAG Builder.

Creates reasoning trace edges from orchestrator query execution traces
with ``projection=["reasoning"]``.

Strategies (deterministic, no LLM calls):
    1. **REASONING_CHAIN** — Orchestrator reasoning steps as a linked chain
    2. **INFERS** — Evidence packet → answer claim (trace context)
    3. **DERIVES_FROM** — Answer → evidence packet that supports it
    4. **USES** — Orchestrator → planner/navigator/critic step

Usage:
    builder = ReasoningDagBuilder(storage)
    edges = await builder.build(trace_events, answer, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
from typing import Any

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import ApexAnswer, EvidencePacket, KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.reasoning_dag")

# Suffixes for synthetic reasoning node IDs
_REASONING_PREFIX = "reasoning:trace"


class ReasoningDagBuilder:
    """Builds ReasoningDAG edges from orchestrator traces and query answers.

    All edges carry ``projection=["reasoning"]`` with reasoning metadata
    including trace_id, step_number, and reasoning_stage.
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        trace_events: list[dict[str, Any]] | None = None,
        answer: ApexAnswer | None = None,
        *,
        doc_id: str,
        query: str = "",
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build ReasoningDAG edges from orchestrator traces and query context.

        Runs up to 4 strategies:
            1. REASONING_CHAIN — ordered step transitions from trace events
            2. DERIVES_FROM — answer claim → evidence packets
            3. INFERS — evidence → claim (inter-packet reasoning)
            4. USES — reasoning step → evidence packet

        Args:
            trace_events: List of trace dicts from ``trace_manager`` events.
            answer:       The :class:`ApexAnswer` produced by the orchestrator.
            doc_id:       The document ID.
            query:        The original user query.
            tenant_id:    Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["reasoning"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        # ── 1. REASONING_CHAIN edges from trace events ──────────────
        reasoning_chain: list[str] = []
        if trace_events:
            for idx, event in enumerate(trace_events):
                event_type = event.get("event", "unknown")
                trace_id = event.get("trace_id", "unknown")
                node_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:step_{idx}"

                reasoning_chain.append(node_id)

                # Link step_{idx} → step_{idx+1}
                if idx > 0:
                    prev_id = reasoning_chain[idx - 1]
                    key = (prev_id, node_id, "REASONING_CHAIN")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=prev_id,
                                target_id=node_id,
                                relation_type=RelationType.REFINES,
                                strength=0.9,
                                evidence=f"Reasoning step {idx}: {event_type}",
                                projections=["reasoning"],
                                metadata={
                                    "step": idx,
                                    "event_type": event_type,
                                    "trace_id": trace_id,
                                    "stage": "orchestration",
                                },
                            ).to_knowledge_edge()
                        )

                # Link trace to the node's data
                for key_field in ("doc_id", "node_id", "source_node"):
                    val = event.get(key_field)
                    if val and isinstance(val, str) and len(val) > 10:
                        edge_key = (node_id, val, "INFERS")
                        if edge_key not in seen:
                            seen.add(edge_key)
                            all_edges.append(
                                GraphEdge(
                                    source_id=node_id,
                                    target_id=val,
                                    relation_type=RelationType.SUPPORTS,
                                    strength=0.8,
                                    evidence=f"Reasoning step {idx} references {key_field}={val[:8]}",
                                    projections=["reasoning"],
                                    metadata={
                                        "step": idx,
                                        "event_type": event_type,
                                        "ref_field": key_field,
                                    },
                                ).to_knowledge_edge()
                            )

        # ── 2. DERIVES_FROM edges (answer → evidence packets) ──────
        if answer and answer.evidence_packets:
            answer_node_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:answer"
            for packet in answer.evidence_packets:
                pkt_node_id = (
                    getattr(packet, "node_id", None)
                    or getattr(getattr(packet, "node", None), "node_id", None)
                )
                if pkt_node_id:
                    key = (answer_node_id, pkt_node_id, "DERIVES_FROM")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=answer_node_id,
                                target_id=pkt_node_id,
                                relation_type=RelationType.DEPENDS_ON,
                                strength=0.9,
                                evidence=f"Answer derived from evidence packet {pkt_node_id[:8]}",
                                projections=["reasoning"],
                                metadata={
                                    "stage": "synthesis",
                                    "query": query[:100] if query else "",
                                },
                            ).to_knowledge_edge()
                        )

                    # 3. INFERS edges (evidence → claim)
                    claim_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:claim:{pkt_node_id[:8]}"
                    key_infers = (pkt_node_id, claim_id, "INFERS")
                    if key_infers not in seen:
                        seen.add(key_infers)
                        all_edges.append(
                            GraphEdge(
                                source_id=pkt_node_id,
                                target_id=claim_id,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.7,
                                evidence=f"Evidence packet {pkt_node_id[:8]} supports claim",
                                projections=["reasoning"],
                                metadata={
                                    "stage": "evidence_to_claim",
                                    "packet_id": pkt_node_id,
                                },
                            ).to_knowledge_edge()
                        )

        # ── 4. USES edges (orchestration components → evidence) ─────
        if answer and answer.evidence_packets:
            for component in ("planner", "navigator", "critic", "synthesizer", "auditor"):
                comp_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:{component}"
                for packet in answer.evidence_packets[:3]:  # Limit to 3 for brevity
                    pkt_node_id = (
                        getattr(packet, "node_id", None)
                        or getattr(getattr(packet, "node", None), "node_id", None)
                    )
                    if pkt_node_id:
                        key = (comp_id, pkt_node_id, "USES")
                        if key not in seen:
                            seen.add(key)
                            all_edges.append(
                                GraphEdge(
                                    source_id=comp_id,
                                    target_id=pkt_node_id,
                                    relation_type=RelationType.REFERENCES,
                                    strength=0.5,
                                    evidence=f"Component '{component}' processed packet {pkt_node_id[:8]}",
                                    projections=["reasoning"],
                                    metadata={
                                        "component": component,
                                        "stage": "orchestration",
                                    },
                                ).to_knowledge_edge()
                            )

        # ── 5. Causality edges from ApexAnswer ──────────────────────
        if answer:
            for idx, edge in enumerate(answer.causal_chain):
                ce_id = f"{_REASONING_PREFIX}:causal:{doc_id[:8]}:{idx}"
                eid = getattr(edge, "edge_id", ce_id)
                src = getattr(edge, "source_node_id", "")
                tgt = getattr(edge, "target_node_id", "")
                etype = getattr(edge, "edge_type", "SUPPORTS")
                if src and tgt:
                    key = (src, tgt, f"CAUSAL_{etype}")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=src,
                                target_id=tgt,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.6,
                                evidence=f"Causal edge from orchestrator: {etype}",
                                projections=["reasoning"],
                                metadata={
                                    "edge_id": eid,
                                    "causal_type": etype,
                                    "stage": "causal_chain",
                                },
                            ).to_knowledge_edge()
                        )

        logger.info(
            "ReasoningDAG: %d edges from doc %s (trace events=%s, packets=%s)",
            len(all_edges),
            doc_id[:8],
            len(trace_events) if trace_events else 0,
            len(answer.evidence_packets) if answer and answer.evidence_packets else 0,
        )
        return all_edges

    async def build_offline(
        self,
        *,
        doc_id: str,
        query: str,
        evidence_packets: list[EvidencePacket],
        causal_edges: list[KnowledgeEdge] | None = None,
    ) -> list[KnowledgeEdge]:
        """Build ReasoningDAG edges without live trace events.

        Useful for offline or post-hoc reasoning graph construction
        when the orchestrator trace is not available.

        Args:
            doc_id:           The document ID.
            query:            The original user query.
            evidence_packets: The evidence packets used in the answer.
            causal_edges:     Optional causal edges from the orchestrator.

        Returns:
            ReasoningDAG edges.
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        query_node_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:query"
        answer_node_id = f"{_REASONING_PREFIX}:{doc_id[:8]}:answer"

        # Query → Answer
        key = (query_node_id, answer_node_id, "INFERS")
        if key not in seen:
            seen.add(key)
            all_edges.append(
                GraphEdge(
                    source_id=query_node_id,
                    target_id=answer_node_id,
                    relation_type=RelationType.SUPPORTS,
                    strength=0.9,
                    evidence=f"Query leads to answer for doc {doc_id[:8]}",
                    projections=["reasoning"],
                    metadata={"query": query[:100], "stage": "query_to_answer"},
                ).to_knowledge_edge()
            )

        if evidence_packets:
            for packet in evidence_packets:
                pkt_node_id = (
                    getattr(packet, "node_id", None)
                    or getattr(getattr(packet, "node", None), "node_id", None)
                )
                if pkt_node_id:
                    # Answer → Evidence
                    key_derives = (answer_node_id, pkt_node_id, "DERIVES_FROM")
                    if key_derives not in seen:
                        seen.add(key_derives)
                        all_edges.append(
                            GraphEdge(
                                source_id=answer_node_id,
                                target_id=pkt_node_id,
                                relation_type=RelationType.DEPENDS_ON,
                                strength=0.9,
                                evidence=f"Answer derived from packet {pkt_node_id[:8]}",
                                projections=["reasoning"],
                                metadata={"stage": "offline_synthesis"},
                            ).to_knowledge_edge()
                        )

        # Causal edges → mirror with reasoning tag
        if causal_edges:
            for ce in causal_edges:
                src = getattr(ce, "source_node_id", "")
                tgt = getattr(ce, "target_node_id", "")
                etype = getattr(ce, "edge_type", "SUPPORTS")
                if src and tgt:
                    key_ce = (src, tgt, f"REASONING_{etype}")
                    if key_ce not in seen:
                        seen.add(key_ce)
                        all_edges.append(
                            GraphEdge(
                                source_id=src,
                                target_id=tgt,
                                relation_type=RelationType.REFERENCES,
                                strength=0.6,
                                evidence=f"Reasoning mirrors causal {etype}",
                                projections=["reasoning"],
                                metadata={"causal_type": etype, "stage": "offline_causal"},
                            ).to_knowledge_edge()
                        )

        logger.info(
            "ReasoningDAG (offline): %d edges from doc %s",
            len(all_edges),
            doc_id[:8],
        )
        return all_edges
