from pydantic import BaseModel, Field

from apex_rag.models.unified_models import CausalEdge, EdgeType, EvidencePacket
from apex_rag.temporal.lineage import DocumentLineageEngine


class ContradictionConflict(BaseModel):
    """Details of a single contradiction conflict between two packets."""

    edge_id: str
    packet_a_id: str
    packet_b_id: str
    evidence_text: str
    resolved_authoritative_id: str
    resolution_reason: str


class ContradictionReport(BaseModel):
    """Full contradiction audit report."""

    has_conflicts: bool = False
    conflicts: list[ContradictionConflict] = Field(default_factory=list)
    authoritative_packets: list[EvidencePacket] = Field(default_factory=list)


class ContradictionResolver:
    """
    Evolved v3 Contradiction Resolver.
    Identifies, logs, and resolves conflicting factual claims in EvidencePackets.
    Ensures contradictory evidence is never synthesized silently.
    """

    def __init__(self, lineage_engine: DocumentLineageEngine | None = None) -> None:
        self.lineage_engine = lineage_engine

    def resolve(
        self,
        packets: list[EvidencePacket],
        contradiction_edges: list[CausalEdge],
    ) -> ContradictionReport:
        """
        Processes EvidencePackets and contradiction edges.
        Determines the latest or most authoritative versions, logs conflicts,
        and returns a resolved packet set with an audit trail report.
        """
        if not contradiction_edges or len(packets) < 2:
            return ContradictionReport(
                has_conflicts=False,
                conflicts=[],
                authoritative_packets=list(packets),
            )

        pkt_map = {p.node_id: p for p in packets}
        resolved_conflicts: list[ContradictionConflict] = []
        suppressed_ids: set[str] = set()

        for edge in contradiction_edges:
            # Only process contradiction edges
            if edge.edge_type != EdgeType.CONTRADICTS:
                continue

            node_a_id = edge.source_node_id
            node_b_id = edge.target_node_id

            if node_a_id not in pkt_map or node_b_id not in pkt_map:
                continue

            pkt_a = pkt_map[node_a_id]
            pkt_b = pkt_map[node_b_id]

            # Determine authoritative source
            authoritative_id = ""
            reason = ""

            # Check if one supersedes the other in lineage engine
            if self.lineage_engine:
                latest_a = self.lineage_engine.resolve_latest_version(pkt_a.document_id)
                latest_b = self.lineage_engine.resolve_latest_version(pkt_b.document_id)
                if latest_a != pkt_a.document_id and latest_b == pkt_b.document_id:
                    authoritative_id = node_b_id
                    reason = f"Document {pkt_a.document_id} is superseded in DocumentLineageEngine."
                elif latest_b != pkt_b.document_id and latest_a == pkt_a.document_id:
                    authoritative_id = node_a_id
                    reason = f"Document {pkt_b.document_id} is superseded in DocumentLineageEngine."

            # Fallback to freshness score comparison
            if not authoritative_id:
                if pkt_a.freshness_score > pkt_b.freshness_score:
                    authoritative_id = node_a_id
                    reason = f"Packet A is fresher ({pkt_a.freshness_score:.3f} > {pkt_b.freshness_score:.3f})."
                elif pkt_b.freshness_score > pkt_a.freshness_score:
                    authoritative_id = node_b_id
                    reason = f"Packet B is fresher ({pkt_b.freshness_score:.3f} > {pkt_a.freshness_score:.3f})."
                else:
                    # Fallback to confidence
                    if pkt_a.confidence >= pkt_b.confidence:
                        authoritative_id = node_a_id
                        reason = f"Packet A has higher confidence ({pkt_a.confidence:.3f} >= {pkt_b.confidence:.3f})."
                    else:
                        authoritative_id = node_b_id
                        reason = f"Packet B has higher confidence ({pkt_b.confidence:.3f} > {pkt_a.confidence:.3f})."

            suppressed_id = node_b_id if authoritative_id == node_a_id else node_a_id
            suppressed_ids.add(suppressed_id)

            conflict = ContradictionConflict(
                edge_id=edge.edge_id,
                packet_a_id=node_a_id,
                packet_b_id=node_b_id,
                evidence_text=edge.evidence,
                resolved_authoritative_id=authoritative_id,
                resolution_reason=reason,
            )
            resolved_conflicts.append(conflict)

        # Filter out suppressed packets
        authoritative_packets = [p for p in packets if p.node_id not in suppressed_ids]

        return ContradictionReport(
            has_conflicts=len(resolved_conflicts) > 0,
            conflicts=resolved_conflicts,
            authoritative_packets=authoritative_packets,
        )
