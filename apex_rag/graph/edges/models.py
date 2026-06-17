"""Edge models for the Causal Knowledge Graph.

Connects the high-level :class:`GraphEdge` (used by builders & retrievers)
with the persistence-ready :class:`CausalEdge` in the unified models.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from apex_rag.models.unified_models import CausalEdge, EdgeType


# ── Extended relation types beyond the base EdgeType ─────────────────


class RelationType(str, Enum):
    """Extended edge types for the Structural Retrieval Graph.

    Includes the base :class:`EdgeType` values plus SRG-specific types
    like ``REFERENCES_TABLE``, ``EXPLAINS``, and ``DEPENDS_ON``.
    """

    # Base causal types
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

    # Structural Retrieval Graph extensions
    REFERENCES_TABLE = "REFERENCES_TABLE"

    @classmethod
    def from_edge_type(cls, et: EdgeType | str) -> RelationType:
        """Convert a base :class:`EdgeType` (or its string value) to the matching :class:`RelationType`."""
        value = et.value if isinstance(et, EdgeType) else et
        return cls(value)


class GraphEdge(BaseModel):
    """A lightweight, serialisable relationship between two ASTNodes.

    This is the high-level edge model used by the :class:`CausalGraphBuilder`
    and :class:`CausalRetriever`.  It mirrors the core :class:`CausalEdge`
    but allows a wider ``relation_type`` vocabulary and carries an optional
    ``metadata`` dict for SRG extensions.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(..., description="ID of the origin ASTNode")
    target_id: str = Field(..., description="ID of the destination ASTNode")
    relation_type: RelationType = Field(
        ..., description="Semantic relationship type"
    )
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Human-readable reasoning")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Converters ─────────────────────────────────────────────────────

    def to_causal_edge(self) -> CausalEdge:
        """Convert this :class:`GraphEdge` to a persistence-ready :class:`CausalEdge`.

        The ``edge_type`` is narrowed to the base :class:`EdgeType`; any
        SRG-specific type (e.g. ``REFERENCES_TABLE``) is stored as
        ``SUPPORTS`` in the core causal layer.
        """
        try:
            causal_type = EdgeType(self.relation_type.value)
        except ValueError:
            # Fall back for SRG-only types
            causal_type = EdgeType.SUPPORTS

        return CausalEdge(
            edge_id=self.id,
            source_node_id=self.source_id,
            target_node_id=self.target_id,
            edge_type=causal_type,
            strength=self.strength,
            evidence=self.evidence,
        )

    @classmethod
    def from_causal_edge(cls, edge: CausalEdge) -> GraphEdge:
        """Wrap an existing :class:`CausalEdge` as a :class:`GraphEdge`."""
        return cls(
            id=edge.edge_id,
            source_id=edge.source_node_id,
            target_id=edge.target_node_id,
            relation_type=RelationType.from_edge_type(edge.edge_type),
            strength=edge.strength,
            evidence=edge.evidence,
        )
