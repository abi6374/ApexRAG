
from pydantic import BaseModel, Field


class SemanticModel(BaseModel):
    """
    Replaces simple string summaries.
    This model provides the LLM with structured 'signposts' to decide navigation.
    """
    node_id: str
    concise_summary: str = Field(..., description="A strict 30-word summary of the node's content.")
    detailed_summary: str | None = Field(None, description="A longer summary for highly complex sections.")
    keywords: list[str] = Field(default_factory=list, description="Extracted semantic keywords.")
    intent_tags: list[str] = Field(default_factory=list, description="Tags like 'factual', 'procedural', 'financial'.")
    entities: list[str] = Field(default_factory=list, description="Named entities found in the text.")
    domain_tags: list[str] = Field(default_factory=list, description="Domain-specific tags (e.g., 'Q3', 'Revenue').")
    confidence_score: float = Field(1.0, description="Confidence in the extraction quality.")
