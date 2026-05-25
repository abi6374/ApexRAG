from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Represents a spatial bounding box for document layout parsing (e.g., from PDFs)."""
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

class ASTNodeMetadata(BaseModel):
    """Metadata associated with an AST Node."""
    page_num: int | None = None
    bounding_box: BoundingBox | None = None
    source_file: str | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

class ASTNode(BaseModel):
    """
    The Universal Document AST Node.
    This represents a structured chunk of a document (e.g., Section, Paragraph, Table).
    """
    id: str
    node_type: str = Field(..., description="e.g., 'Section', 'Paragraph', 'Table', 'ListItem'")
    content: str = Field(..., description="The literal text content of the node")
    metadata: ASTNodeMetadata = Field(default_factory=ASTNodeMetadata)
    children: list['ASTNode'] = Field(default_factory=list)

    # Optional reference to a parent (can be useful for graph traversals)
    parent_id: str | None = None

    def add_child(self, child: 'ASTNode') -> None:
        self.children.append(child)
        child.parent_id = self.id

# Needed for self-referencing model validation in Pydantic v2 (or update_forward_refs in v1)
ASTNode.model_rebuild()
