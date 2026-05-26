from pydantic import BaseModel, Field


class EvidencePacket(BaseModel):
    """
    Strongly typed evidence payload. This is the ONLY valid input for the Synthesizer.
    It guarantees that all answers are grounded in verifiable, structured provenance.
    """

    node_id: str = Field(..., description="ID of the verified AST Node")
    source_document: str = Field(..., description="Original document name/ID")
    section_path: str = Field(..., description="Logical path (e.g. 'Chapter 1 > Sec 2')")
    page_number: int | None = Field(None, description="Page number for exact citation")
    paragraph_index: int | None = Field(None, description="Position within the section")
    bounding_box: str | None = Field(None, description="Spatial coordinates for PDF highlighting")

    retrieval_reason: str = Field(
        ..., description="Why this node was selected (from the Planner/Navigator)"
    )
    verification_result: bool = Field(..., description="Must be True to be synthesized")
    supporting_nodes: list[str] = Field(default_factory=list, description="IDs of context nodes")
    graph_relationships: list[dict[str, str]] = Field(
        default_factory=list, description="Relevant SRG edges"
    )

    confidence_score: float = Field(..., description="Mathematical score of retrieval confidence")
    provenance_chain: list[str] = Field(
        default_factory=list, description="Trace of agents involved"
    )
    contradiction_flags: list[str] = Field(
        default_factory=list, description="Detected conflicts with other packets"
    )

    content: str = Field(..., description="The literal text to be synthesized")
