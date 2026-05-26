from typing import Any

from pydantic import BaseModel, Field


class GraphEdge(BaseModel):
    """
    Represents a relationship (edge) between two ASTNodes in the Structural Retrieval Graph.
    """

    id: int | None = None
    source_id: str = Field(..., description="ID of the origin ASTNode")
    target_id: str = Field(..., description="ID of the destination ASTNode")
    relation_type: str = Field(
        ..., description="E.g., 'REFERENCES_TABLE', 'EXPLAINS', 'DEPENDS_ON'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
