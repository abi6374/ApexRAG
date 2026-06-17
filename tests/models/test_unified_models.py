"""Tests for the ApexRAG unified data models (Part 1).

Covers all five core types:
    - ASTNode
    - TemporalMetadata
    - CausalEdge
    - EvidencePacket
    - ApexAnswer

Test categories (15 total):
    1. Field type validation      (tests 1–5)
    2. Default values             (tests 6–8)
    3. JSON round-trip            (tests 9–11)
    4. Edge cases                 (tests 12–14)
    5. Enum validation            (test  15)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from apex_rag.models.unified_models import (
    ASTNode,
    ApexAnswer,
    CausalEdge,
    EdgeType,
    EvidencePacket,
    NodeType,
    TemporalMetadata,
)


# ═══════════════════════════════════════════════════════════
# 1. Field type validation — 5 tests
# ═══════════════════════════════════════════════════════════


class TestASTNodeFieldTypes:
    """Verify that ASTNode rejects invalid types for each field."""

    def test_invalid_node_id_rejected(self) -> None:
        """node_id must be a valid UUID4 string."""
        with pytest.raises(ValidationError):
            ASTNode(node_id="not-a-uuid", content="x", node_type=NodeType.PARAGRAPH, doc_id="d1")

    def test_invalid_node_type_rejected(self) -> None:
        """node_type must be a member of NodeType enum."""
        with pytest.raises(ValidationError):
            # noinspection PyTypeChecker
            ASTNode(node_id=str(uuid.uuid4()), content="x", node_type="INVALID", doc_id="d1")  # type: ignore[arg-type]

    def test_negative_depth_rejected(self) -> None:
        """depth must be >= 0."""
        with pytest.raises(ValidationError):
            ASTNode(
                node_id=str(uuid.uuid4()),
                content="x",
                node_type=NodeType.HEADING,
                doc_id="d1",
                depth=-1,
            )

    def test_non_string_content_rejected(self) -> None:
        """content must be a string."""
        with pytest.raises(ValidationError):
            ASTNode(
                node_id=str(uuid.uuid4()),
                content=42,  # type: ignore[arg-type]
                node_type=NodeType.CODE,
                doc_id="d1",
            )

    def test_automatic_uuid_generation(self) -> None:
        """When node_id is omitted, a valid UUID4 is auto-generated."""
        node = ASTNode(content="hello", node_type=NodeType.LIST, doc_id="docs/readme.md")
        # Verify it's a valid UUID4
        parsed = uuid.UUID(node.node_id, version=4)
        assert str(parsed) == node.node_id


# ═══════════════════════════════════════════════════════════
# 2. Default values — 3 tests
# ═══════════════════════════════════════════════════════════


class TestDefaultValues:
    """Verify sensible defaults across all models."""

    def test_ast_node_defaults(self) -> None:
        """depth=0, parent_id=None, children=[], source_date=None, embedding=[]. """
        node = ASTNode(content="Hello", node_type=NodeType.PARAGRAPH, doc_id="doc1")
        assert node.depth == 0
        assert node.parent_id is None
        assert node.children == []
        assert node.source_date is None
        assert node.embedding == []

    def test_temporal_metadata_defaults(self) -> None:
        """freshness_score=1.0, decay_rate=0.001, superseded_by=None."""
        meta = TemporalMetadata(node_id=str(uuid.uuid4()))
        assert meta.freshness_score == 1.0
        assert meta.decay_rate == 0.001
        assert meta.superseded_by is None

    def test_causal_edge_defaults(self) -> None:
        """strength=0.5, evidence='', edge auto-generated."""
        edge = CausalEdge(
            source_node_id=str(uuid.uuid4()),
            target_node_id=str(uuid.uuid4()),
            edge_type=EdgeType.SUPPORTS,
        )
        assert edge.strength == 0.5
        assert edge.evidence == ""
        assert edge.discovered_at is not None
        parsed = uuid.UUID(edge.edge_id, version=4)
        assert str(parsed) == edge.edge_id


# ═══════════════════════════════════════════════════════════
# 3. JSON round-trip serialisation — 3 tests
# ═══════════════════════════════════════════════════════════


class TestJSONRoundTrip:
    """Each model must survive a model_dump → json → model_validate cycle."""

    @staticmethod
    def _round_trip(model: BaseModel) -> BaseModel:
        data = model.model_dump(mode="json")
        raw_json = json.dumps(data)
        restored = model.__class__.model_validate(json.loads(raw_json))
        return restored

    def test_ast_node_round_trip(self) -> None:
        """ASTNode with all fields populated survives a round trip."""
        original = ASTNode(
            node_id=str(uuid.uuid4()),
            content="## Section 1\n\nThis is a heading section.",
            node_type=NodeType.HEADING,
            depth=2,
            parent_id=str(uuid.uuid4()),
            children=[str(uuid.uuid4()), str(uuid.uuid4())],
            doc_id="report.pdf",
            source_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ingestion_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
            embedding=[0.1, 0.2, 0.3],
        )
        restored = self._round_trip(original)
        assert restored.node_id == original.node_id
        assert restored.content == original.content
        assert restored.node_type == original.node_type
        assert restored.depth == original.depth
        assert restored.parent_id == original.parent_id
        assert restored.children == original.children
        assert restored.doc_id == original.doc_id
        assert restored.source_date == original.source_date
        assert restored.ingestion_date == original.ingestion_date
        assert restored.embedding == pytest.approx(original.embedding)

    def test_causal_edge_round_trip(self) -> None:
        """CausalEdge with CONTADICTS type survives a round trip."""
        original = CausalEdge(
            edge_id=str(uuid.uuid4()),
            source_node_id=str(uuid.uuid4()),
            target_node_id=str(uuid.uuid4()),
            edge_type=EdgeType.CONTRADICTS,
            strength=0.87,
            evidence="Node A says Q3 revenue is $40M, Node B says $52M.",
            discovered_at=datetime(2025, 3, 10, tzinfo=timezone.utc),
        )
        restored = self._round_trip(original)
        assert restored.edge_id == original.edge_id
        assert restored.source_node_id == original.source_node_id
        assert restored.target_node_id == original.target_node_id
        assert restored.edge_type == original.edge_type
        assert restored.strength == pytest.approx(original.strength)
        assert restored.evidence == original.evidence

    def test_apex_answer_round_trip(self) -> None:
        """Full ApexAnswer with nested EvidencePackets survives a round trip."""
        node = ASTNode(
            node_id=str(uuid.uuid4()),
            content="Q3 revenue was $52M.",
            node_type=NodeType.PARAGRAPH,
            doc_id="report_2024.pdf",
        )
        temporal = TemporalMetadata(node_id=node.node_id, freshness_score=0.95)
        edge = CausalEdge(
            source_node_id=node.node_id,
            target_node_id=str(uuid.uuid4()),
            edge_type=EdgeType.SUPPORTS,
        )
        packet = EvidencePacket(
            node=node,
            temporal_metadata=temporal,
            causal_edges=[edge],
            retrieval_score=0.92,
            nonconformity_score=0.12,
            rank=1,
        )
        answer = ApexAnswer(
            answer_text="Q3 revenue was $52M [Node ID: {}].".format(node.node_id),
            evidence_packets=[packet],
            temporal_freshness=0.95,
            contradictions=[],
            coverage_guarantee=0.90,
            prediction_set_size=1,
            query="What is Q3 revenue?",
            latency_ms=1423.5,
        )
        restored = self._round_trip(answer)
        assert restored.answer_text == answer.answer_text
        assert restored.temporal_freshness == answer.temporal_freshness
        assert restored.coverage_guarantee == answer.coverage_guarantee
        assert restored.prediction_set_size == answer.prediction_set_size
        assert restored.query == answer.query
        assert restored.latency_ms == answer.latency_ms
        # Nested EvidencePacket
        assert len(restored.evidence_packets) == 1
        rp = restored.evidence_packets[0]
        assert rp.node.content == node.content
        assert rp.temporal_metadata.freshness_score == temporal.freshness_score
        assert rp.retrieval_score == 0.92
        assert rp.nonconformity_score == 0.12
        assert rp.rank == 1
        assert len(rp.causal_edges) == 1
        assert rp.causal_edges[0].edge_type == EdgeType.SUPPORTS


# ═══════════════════════════════════════════════════════════
# 4. Edge cases — 3 tests
# ═══════════════════════════════════════════════════════════


class TestEdgeCases:
    """Verify behaviour at the boundaries of model constraints."""

    def test_empty_children_list(self) -> None:
        """An ASTNode with an empty children list is valid."""
        node = ASTNode(content="Leaf node", node_type=NodeType.PARAGRAPH, doc_id="d1", children=[])
        assert node.children == []

    def test_none_source_date(self) -> None:
        """source_date=None is valid for both ASTNode and TemporalMetadata."""
        node = ASTNode(content="No date", node_type=NodeType.CODE, doc_id="d1", source_date=None)
        assert node.source_date is None

        meta = TemporalMetadata(node_id=node.node_id, source_date=None)
        assert meta.source_date is None

    def test_zero_length_evidence_list(self) -> None:
        """EvidencePacket with no causal_edges is valid."""
        node = ASTNode(content="Standalone", node_type=NodeType.LIST, doc_id="d1")
        meta = TemporalMetadata(node_id=node.node_id)
        packet = EvidencePacket(node=node, temporal_metadata=meta, causal_edges=[])
        assert packet.causal_edges == []


# ═══════════════════════════════════════════════════════════
# 5. Enum validation — 2 tests
# ═══════════════════════════════════════════════════════════


class TestEnumValidation:
    """Verify that only valid enum members are accepted."""

    def test_all_node_types_constructible(self) -> None:
        """Every NodeType value can be used to build an ASTNode."""
        for nt in NodeType:
            node = ASTNode(content=f"test-{nt.value}", node_type=nt, doc_id="d1")
            assert node.node_type == nt

    def test_invalid_edge_type_rejected(self) -> None:
        """Non-EdgeType strings are rejected by CausalEdge."""
        with pytest.raises(ValidationError):
            CausalEdge(
                source_node_id=str(uuid.uuid4()),
                target_node_id=str(uuid.uuid4()),
                edge_type="INVALID_TYPE",  # type: ignore[arg-type]
            )

    def test_all_edge_types_constructible(self) -> None:
        """Every EdgeType value can be used to build a CausalEdge."""
        sid = str(uuid.uuid4())
        tid = str(uuid.uuid4())
        for et in EdgeType:
            edge = CausalEdge(source_node_id=sid, target_node_id=tid, edge_type=et)
            assert edge.edge_type == et


# ═══════════════════════════════════════════════════════════
# 6. Additional sanity checks — 1 test
# ═══════════════════════════════════════════════════════════


class TestApexAnswerDefaults:
    """Defaults and bounds for ApexAnswer."""

    def test_apex_answer_defaults(self) -> None:
        """ApexAnswer sensible defaults."""
        ans = ApexAnswer(answer_text="Test answer.")
        assert ans.evidence_packets == []
        assert ans.temporal_freshness == 0.0
        assert ans.contradictions == []
        assert ans.coverage_guarantee == 0.0
        assert ans.prediction_set_size == 0
        assert ans.causal_chain == []
        assert ans.query == ""
        assert ans.latency_ms == 0.0
