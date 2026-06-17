"""
agents/audit/temporal_audit.py — TemporalAuditAgent for reviewing evidence conflicts.

The TemporalAuditAgent examines a set of EvidencePackets and identifies any
temporal contradictions between them.  It uses the TemporalContradictionDetector
and FreshnessScorer to produce a list of conflict CausalEdges.

Usage::

    auditor = TemporalAuditAgent(
        contradiction_detector=detector,
        freshness_scorer=scorer,
    )
    conflicts = await auditor.audit(packets)
    # => [CausalEdge(type=CONTRACTS, ...), ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from apex_rag.models.unified_models import (
    ASTNode,
    CausalEdge,
    EvidencePacket,
    NodeType,
)
from apex_rag.temporal.contradiction import TemporalContradictionDetector
from apex_rag.temporal.scorer import FreshnessScorer

logger = logging.getLogger("apex_rag.agents.audit.temporal")


@dataclass
class AuditReport:
    """Summary of the temporal audit for a query.

    Attributes:
        passed:        True if no contradictions were found.
        conflicts:     List of contradiction CausalEdges discovered.
        mean_freshness: Average freshness score across all evidence.
        stale_packets: List of (packet, freshness) tuples where
                       freshness < 0.1 (outdated evidence).
    """

    passed: bool
    conflicts: list[CausalEdge]
    mean_freshness: float
    stale_packets: list[tuple[EvidencePacket, float]]


class TemporalAuditAgent:
    """Reviews evidence packets for temporal conflicts and freshness issues.

    For a given set of EvidencePackets (from the retriever), this agent:

    1. Converts packets to unified ASTNodes for the contradiction detector.
    2. Runs :class:`TemporalContradictionDetector` to find CONTRADICTS edges.
    3. Computes freshness scores for each packet via :class:`FreshnessScorer`.
    4. Flags packets with freshness < 0.1 as stale.
    5. Returns an :class:`AuditReport` summarising the findings.

    The Orchestrator should call this agent after retrieval and before
    synthesis, and surface any contradictions in the ApexAnswer.
    """

    def __init__(
        self,
        contradiction_detector: TemporalContradictionDetector | None = None,
        freshness_scorer: FreshnessScorer | None = None,
    ) -> None:
        self._detector = contradiction_detector or TemporalContradictionDetector()
        self._scorer = freshness_scorer or FreshnessScorer(domain="general")

    # ── Public API ────────────────────────────────────────────────────

    async def audit(
        self,
        packets: list[EvidencePacket],
        *,
        doc_id: str = "unknown",
    ) -> AuditReport:
        """Run a temporal audit over the given evidence packets.

        Args:
            packets: The evidence packets to audit (from retrieval).
            doc_id:  Document ID for constructing ASTNode references.

        Returns:
            An :class:`AuditReport` with conflicts, freshness, and staleness info.
        """
        if not packets:
            return AuditReport(
                passed=True,
                conflicts=[],
                mean_freshness=1.0,
                stale_packets=[],
            )

        # 1. Convert to unified ASTNodes for the contradiction detector
        nodes = self._packets_to_nodes(packets, doc_id)

        # 2. Detect contradictions
        conflicts: list[CausalEdge] = []
        if len(packets) >= 2:
            conflicts = await self._detector.detect_all(nodes)
            logger.info(
                "TemporalAudit: found %d contradiction(s) across %d packets",
                len(conflicts),
                len(packets),
            )

        # 3. Score freshness
        freshness_scores: list[float] = []
        stale: list[tuple[EvidencePacket, float]] = []
        for pkt, node in zip(packets, nodes, strict=False):
            score = self._scorer.compute(node.source_date)
            freshness_scores.append(score)
            if score < 0.1:
                stale.append((pkt, score))

        mean_freshness = (
            sum(freshness_scores) / len(freshness_scores)
            if freshness_scores
            else 1.0
        )

        logger.info(
            "TemporalAudit: mean_freshness=%.4f, stale=%d/%d",
            mean_freshness,
            len(stale),
            len(packets),
        )

        return AuditReport(
            passed=len(conflicts) == 0,
            conflicts=conflicts,
            mean_freshness=mean_freshness,
            stale_packets=stale,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _packets_to_nodes(
        packets: list[EvidencePacket],
        doc_id: str,
    ) -> list[ASTNode]:
        """Convert EvidencePackets to unified ASTNodes for contradiction detection.

        Extracts source_date from the packet's temporal_metadata if available,
        or from the evidence's node.source_date.
        """
        nodes: list[ASTNode] = []
        for pkt in packets:
            source_date = None
            if pkt.temporal_metadata is not None:
                source_date = pkt.temporal_metadata.source_date
            elif pkt.node is not None:
                source_date = pkt.node.source_date

            nodes.append(
                ASTNode(
                    node_id=pkt.node.node_id if pkt.node else "unknown",
                    node_type=NodeType.PARAGRAPH,
                    content=pkt.node.content if pkt.node else "",
                    doc_id=doc_id,
                    source_date=source_date,
                )
            )
        return nodes
