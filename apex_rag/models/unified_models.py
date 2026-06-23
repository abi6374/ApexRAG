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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    REPLACED_BY = "REPLACED_BY"
    VALID_DURING = "VALID_DURING"
    EFFECTIVE_DURING = "EFFECTIVE_DURING"
    SNAPSHOT_OF = "SNAPSHOT_OF"
    HISTORICAL_PARENT = "HISTORICAL_PARENT"



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

    model_config = ConfigDict(use_enum_values=True)


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

    # Temporal Intelligence extensions
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_from: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_to: datetime | None = None
    version_number: int = 1
    revision_number: int = 0
    source_timestamp: datetime | None = None
    approval_timestamp: datetime | None = Field(default=None, description="When this version was officially approved")
    is_current: bool = True
    superseded_by: str | None = None
    previous_version: str | None = None
    validity_status: str = Field(default="ACTIVE", description="Validity status: ACTIVE, PENDING, EXPIRED, SUPERSEDED, DRAFT, ARCHIVED")

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

    model_config = ConfigDict()


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

    model_config = ConfigDict(use_enum_values=True)


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

    model_config = ConfigDict()



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

    model_config = ConfigDict()


EvidencePacket.model_rebuild()
ApexAnswer.model_rebuild()


# ─────────────────────────────────────────────────────────────
# Temporal Node Version (temporal/temporal_models.py candidate)
# ─────────────────────────────────────────────────────────────


class TemporalNodeVersion(BaseModel):
    """A specific versioned snapshot of an ASTNode with full temporal context.

    Captures every mutation to a node as an immutable version record,
    enabling time-travel queries, audit trails, and lineage tracking.
    """

    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str
    content: str = ""
    doc_id: str
    tenant_id: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_to: datetime | None = None
    version_number: int = 1
    revision_number: int = 0
    source_timestamp: datetime | None = None
    approval_timestamp: datetime | None = None
    is_current: bool = True
    superseded_by: str | None = None
    previous_version: str | None = None
    validity_status: str = "ACTIVE"

    model_config = ConfigDict()


class NodeVersionHistory(BaseModel):
    """Complete version history for a single node, ordered by version number.

    Provides a full audit trail of every modification to a node over time,
    with pointers to previous and superseding versions.
    """

    node_id: str
    doc_id: str
    versions: list[TemporalNodeVersion] = Field(default_factory=list)
    current_version: TemporalNodeVersion | None = None
    total_versions: int = 0

    model_config = ConfigDict()


class VersionLineage(BaseModel):
    """Tracks the lineage chain of a node across supersession events.

    Forms a directed acyclic graph (DAG) showing how a node evolved
    through versions A → B → C, with supersession metadata at each step.
    """

    lineage_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str
    doc_id: str
    tenant_id: str = "default"
    source_version_id: str
    target_version_id: str | None = None
    lineage_type: str = "VERSION_OF"  # VERSION_OF, SUPERSEDES, REPLACED_BY
    strength: float = 1.0
    evidence: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict()


# ─────────────────────────────────────────────────────────────
# Permission Models (Enterprise RBAC)
# ─────────────────────────────────────────────────────────────


class Permission(BaseModel):
    """Base permission model representing an allow/deny rule."""

    permission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    action: str  # read, write, delete, traverse, audit
    is_allowed: bool = True
    priority: int = 0  # Higher priority overrides lower

    model_config = ConfigDict()


class ResourcePermission(Permission):
    """Permission scoped to a specific resource type."""

    resource_type: str  # document, node, version, field


class NodePermission(ResourcePermission):
    """Permission scoped to a specific AST node."""

    node_id: str | None = None  # None = applies to all nodes of this type
    node_type_filter: str | None = None


class DocumentPermission(ResourcePermission):
    """Permission scoped to a specific document."""

    doc_id: str | None = None


class FieldPermission(Permission):
    """Field-level security: controls visibility of specific fields in content."""

    resource_type: str = "ASTNode"
    field_name: str


# ─────────────────────────────────────────────────────────────
# Access Audit Record
# ─────────────────────────────────────────────────────────────


class AccessAuditRecord(BaseModel):
    """Detailed audit record for every access attempt.

    Captures who accessed what, when, which nodes were
    allowed/blocked, and the retrieval mode used.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    role: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str = ""
    accessed_nodes: list[str] = Field(default_factory=list)
    blocked_nodes: list[str] = Field(default_factory=list)
    retrieval_mode: str = ""
    temporal_as_of: datetime | None = None
    allowed: bool = True
    duration_ms: float = 0.0

    model_config = ConfigDict()

