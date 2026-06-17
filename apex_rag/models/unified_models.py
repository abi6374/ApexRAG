"""Unified data models for the ApexRAG four-layer architecture.

Every layer — AST Core, Temporal Intelligence, Causal Knowledge Graph,
and Conformal Uncertainty Quantification — shares these canonical types.
All models are Pydantic BaseModels for automatic JSON serialization,
validation, and schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────


class NodeType(str, Enum):
    """Typed AST node categories that govern how the Navigator
    decides to descend into a branch."""

    PARAGRAPH = "PARAGRAPH"
    HEADING = "HEADING"
    TABLE = "TABLE"
    CODE = "CODE"
    LIST = "LIST"
    IMAGE = "IMAGE"


class EdgeType(str, Enum):
    """Typed relationships in the Causal Knowledge Graph."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    OVERRIDES = "OVERRIDES"
    REFINES = "REFINES"
    DEPENDS_ON = "DEPENDS_ON"
    REFERENCES = "REFERENCES"
    EXPLAINS = "EXPLAINS"
    SAME_TOPIC = "SAME_TOPIC"
    IMPLEMENTS = "IMPLEMENTS"
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    SUCCESSOR = "SUCCESSOR"
    PREDECESSOR = "PREDECESSOR"
    VERSION_OF = "VERSION_OF"
    SUPERSEDES = "SUPERSEDES"



# ─────────────────────────────────────────────────────────────
# Core AST Node
# ─────────────────────────────────────────────────────────────


class ASTNode(BaseModel):
    """A single, addressable node in the Universal Document AST.

    Every paragraph, heading, table, code block, and list item
    becomes an ASTNode.  The tree structure is captured by
    ``parent_id`` and ``children`` (lists of child node IDs or objects).

    Attributes:
        node_id:         Globally unique identifier (UUID4 string).
        content:         Raw text content of the node.
        node_type:       Semantic type from the :class:`NodeType` enum.
        depth:           Depth in the AST tree (0 = root).
        parent_id:       ID of the parent node, or ``None`` for root.
        children:        List of child node IDs or ASTNode instances.
        doc_id:          ID of the source document.
        source_date:     When the source document was authored /
                         published, if known.
        ingestion_date:  When this node was ingested into the system.
        embedding:       Dense vector embedding of ``content``
                         (may be empty until the embedding pass).
    """

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    node_type: NodeType
    depth: int = 0
    parent_id: str | None = None
    children: list[str | ASTNode] = Field(default_factory=list)
    doc_id: str
    source_date: datetime | None = None
    ingestion_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    page_number: int | None = Field(default=None, description="Page number in the source document (1-based)")
    embedding: list[float] = Field(default_factory=list)
    image_data: str | None = Field(default=None, description="Base64-encoded image data URI for IMAGE nodes")

    @field_validator("node_id")
    @classmethod
    def _validate_node_id(cls, v: str) -> str:
        """Ensure node_id is a valid UUID4 string."""
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"node_id must be a valid UUID4 string, got {v!r}")
        return v

    @field_validator("depth")
    @classmethod
    def _depth_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"depth must be >= 0, got {v}")
        return v

    @field_validator("embedding")
    @classmethod
    def _embedding_non_empty(cls, v: list[float]) -> list[float]:
        # embedding is allowed to be empty (pre-embedding pass)
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}


ASTNode.model_rebuild()


# ─────────────────────────────────────────────────────────────
# Temporal Metadata
# ─────────────────────────────────────────────────────────────


class TemporalMetadata(BaseModel):
    """Temporal context attached to every ASTNode.

    Enables freshness-aware retrieval and contradiction detection
    across time periods.

    Attributes:
        node_id:         ID of the associated ASTNode.
        source_date:     When the source document was authored, if
                         known.
        ingestion_date:  When the node was ingested into the system.
        freshness_score: Computed freshness in [0, 1], where 1 = just
                         published, 0 = completely decayed.
        decay_rate:      Per‑document or per‑domain decay rate.
                         Default 0.001 (general documents).
        superseded_by:   If this node has been overridden by a newer
                         node, the ``node_id`` of the overriding node.
    """

    node_id: str
    source_date: datetime | None = None
    ingestion_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_rate: float = 0.001
    superseded_by: str | None = None

    # Temporal Intelligence extensions
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_from: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_to: datetime | None = None
    version_number: int = 1
    revision_number: int = 0
    source_timestamp: datetime | None = None
    is_current: bool = True
    previous_version: str | None = None

    @field_validator("node_id")
    @classmethod
    def _validate_node_id(cls, v: str) -> str:
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"node_id must be a valid UUID4 string, got {v!r}")
        return v

    @field_validator("decay_rate")
    @classmethod
    def _decay_rate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"decay_rate must be >= 0, got {v}")
        return v

    @field_validator("freshness_score")
    @classmethod
    def _freshness_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"freshness_score must be in [0, 1], got {v}")
        return v

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


# ─────────────────────────────────────────────────────────────
# Causal Edge
# ─────────────────────────────────────────────────────────────


class CausalEdge(BaseModel):
    """A typed, directed relationship between two ASTNodes.

    Edges form the Causal Knowledge Graph that the CausalRetriever
    traverses to build answer chains.

    Attributes:
        edge_id:         Globally unique identifier (UUID4 string).
        source_node_id:  Origin node of the directed edge.
        target_node_id:  Destination node of the directed edge.
        edge_type:       Semantic type from the :class:`EdgeType` enum.
        strength:        Confidence in this relationship, in [0, 1].
        evidence:        Human‑readable reasoning for why this edge
                         was created (e.g. LLM justification).
        discovered_at:   Timestamp of edge creation.
    """

    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("edge_id")
    @classmethod
    def _validate_edge_id(cls, v: str) -> str:
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"edge_id must be a valid UUID4 string, got {v!r}")
        return v

    @field_validator("source_node_id", "target_node_id")
    @classmethod
    def _validate_node_id_ref(cls, v: str) -> str:
        try:
            uuid.UUID(v, version=4)
        except ValueError:
            raise ValueError(f"Node reference must be a valid UUID4 string, got {v!r}")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}


# ─────────────────────────────────────────────────────────────
# Evidence Packet
# ─────────────────────────────────────────────────────────────


class EvidencePacket(BaseModel):
    """A fully annotated piece of evidence for the Synthesizer.

    Only verified EvidencePackets are passed to the SynthesizerAgent.
    """

    node_id: str = ""
    document_id: str = ""
    tenant_id: str = "default"
    page_number: int | None = None
    section_path: str = ""
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    graph_context: dict[str, Any] = Field(default_factory=dict)
    reasoning_trace: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    content: str = ""

    # Legacy compatibility fields
    node: ASTNode | None = None
    temporal_metadata: TemporalMetadata | None = None
    causal_edges: list[CausalEdge] = Field(default_factory=list)
    nonconformity_score: float = Field(default=1.0, ge=0.0)
    rank: int = 0

    @model_validator(mode="before")
    @classmethod
    def _populate_from_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Extract fields from legacy node if present
            node = data.get("node")
            if node:
                node_id = getattr(node, "node_id", None) or (node.get("node_id") if isinstance(node, dict) else None)
                doc_id = getattr(node, "doc_id", None) or (node.get("doc_id") if isinstance(node, dict) else None)
                content = getattr(node, "content", None) or (node.get("content") if isinstance(node, dict) else None)
                page_number = getattr(node, "page_number", None) or (node.get("page_number") if isinstance(node, dict) else None)
                path = getattr(node, "path", None) or (node.get("path") if isinstance(node, dict) else None)

                if node_id and not data.get("node_id"):
                    data["node_id"] = node_id
                if doc_id and not data.get("document_id"):
                    data["document_id"] = doc_id
                if content and not data.get("content"):
                    data["content"] = content
                if page_number is not None and data.get("page_number") is None:
                    data["page_number"] = page_number
                if path and not data.get("section_path"):
                    data["section_path"] = path

            # Extract fields from legacy temporal_metadata if present
            temporal_meta = data.get("temporal_metadata")
            if temporal_meta:
                freshness = getattr(temporal_meta, "freshness_score", None) or (temporal_meta.get("freshness_score") if isinstance(temporal_meta, dict) else None)
                if freshness is not None and data.get("freshness_score") is None:
                    data["freshness_score"] = freshness

            # Default node_id and document_id if not present
            if not data.get("node_id"):
                data["node_id"] = ""
            if not data.get("document_id"):
                data["document_id"] = ""

        return data

    @property
    def doc_id(self) -> str:
        return self.document_id

    @field_validator("rank")
    @classmethod
    def _rank_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"rank must be >= 0, got {v}")
        return v

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}



# ─────────────────────────────────────────────────────────────
# Apex Answer
# ─────────────────────────────────────────────────────────────


class ApexAnswer(BaseModel):
    """The final output of the ApexRAG pipeline.

    Contains the answer text, all supporting evidence, temporal
    freshness summary, detected contradictions, and a conformal
    coverage guarantee.

    Attributes:
        answer_text:         The generated answer string, with inline
                             citations ``[Node ID: xxx]``.
        evidence_packets:    The full set of verified evidence packets
                             that support the answer.
        temporal_freshness:  Mean freshness score across all evidence
                             packets (0 = all stale, 1 = all current).
        contradictions:      Only ``CONTRADICTS`` edges that were
                             flagged during the temporal audit.
        coverage_guarantee:  The conformal prediction coverage level,
                             e.g. 0.90 means the answer is provably
                             correct at least 90 % of the time on
                             held‑out queries.
        prediction_set_size: Number of evidence packets in the
                             conformal prediction set.
        causal_chain:        Ordered list of causal edges that link
                             the evidence packets into a reasoning
                             chain.
        query:               The original user query.
        latency_ms:          End‑to‑end pipeline latency in
                             milliseconds.
    """

    answer_text: str
    evidence_packets: list[EvidencePacket] = Field(default_factory=list)
    temporal_freshness: float = Field(default=0.0, ge=0.0, le=1.0)
    contradictions: list[CausalEdge] = Field(default_factory=list)
    coverage_guarantee: float = Field(default=0.0, ge=0.0, le=1.0)
    prediction_set_size: int = 0
    causal_chain: list[CausalEdge] = Field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0

    @field_validator("prediction_set_size")
    @classmethod
    def _size_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"prediction_set_size must be >= 0, got {v}")
        return v

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


EvidencePacket.model_rebuild()
ApexAnswer.model_rebuild()

