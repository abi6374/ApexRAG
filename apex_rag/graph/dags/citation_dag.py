"""
graph/dags/citation_dag.py — CitationDAG Builder.

Extracts citation and cross-reference relationships from AST nodes,
then creates typed KnowledgeEdges with ``projection=["citation"]``.

Strategies (deterministic, no LLM calls):
    1. CROSS_REFERENCE — Internal section references (``see §3.2``, ``cf. Section 5``)
    2. CITES           — Bibliography markers (``[1]``, ``(Smith, 2020)``)
    3. REGULATES       — Regulation references (GDPR, HIPAA, etc.)
    4. ATTRIBUTES      — Entity co-occurrence (FTS5-based cross-document)

Usage:
    builder = CitationDagBuilder(storage)
    edges = await builder.build(nodes, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
import re

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import ASTNode, KnowledgeEdge, NodeType

logger = logging.getLogger("apex_rag.graph.dags.citation_dag")

# ── Citation regex patterns ──────────────────────────────────────────────

# Internal cross-references: "see §3.2", "cf. Section 5.1", "as mentioned in Chapter 4"
_RE_XREF = re.compile(
    r"(?:see|cf\.|refer\s+to|as\s+mentioned\s+in|per|vide)\s+"
    r"((?:§|Section|Sec\.|Chapter|Ch\.|Appendix|Fig\.|Table|Figure)"
    r"\s*\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Bibliography/note markers: [1], [2-4], [Smith2020]
_RE_BIBLIO_MARKER = re.compile(r"(?:^|\s)\[(\d+(?:[-,]\d+)*)\](?:[^a-zA-Z]|$)", re.MULTILINE)

# Parenthetical citations: (Smith et al., 2020), (Doe, 2019)
_RE_PAREN_CITE = re.compile(
    r"\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?),\s*(\d{4})\)"
)

# Regulation references (shared with EntityDAG)
_RE_REGULATION = re.compile(
    r"\b(GDPR|HIPAA|SOX|PCI(?:\sDSS)?|ISO\s*\d+|FASB|IFRS|GAAP|"
    r"ESG|KYC|AML|CCPA|ADA|OSHA|EPA)\b",
    re.IGNORECASE,
)

# Page references: "see p.15", "cf. pp.22-25"
_RE_PAGE_REF = re.compile(
    r"(?:see|cf\.|p\.|page|pp\.)\s*(\d+)(?:\s*[-–]\s*(\d+))?",
    re.IGNORECASE,
)


class CitationDagBuilder:
    """Builds CitationDAG edges from AST nodes.

    All produced edges carry ``projection=["citation"]`` and are tagged
    with citation metadata (ref_text, strategy, page_number).
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,  # noqa: ARG002
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build CitationDAG edges for a document.

        Runs 3 strategies:
            1. CROSS_REFERENCE — internal section references via regex
            2. CITES           — bibliography markers and parenthetical citations
            3. REGULATES       — regulation references via regex

        Args:
            nodes:    AST nodes for the document.
            doc_id:   The document ID (for cross-doc resolution, reserved).
            tenant_id: Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["citation"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        # Build heading map for internal resolution
        headings: dict[str, list[ASTNode]] = {}  # normalized title → nodes
        for node in nodes:
            if node.node_type == NodeType.HEADING:
                norm = node.content.lower().strip()
                headings.setdefault(norm, []).append(node)

        # 1. CROSS_REFERENCE edges
        for node in nodes:
            for match in _RE_XREF.finditer(node.content):
                ref_text = match.group(1).strip()
                target_id = self._resolve_internal_ref(ref_text, headings)
                if target_id:
                    key = (node.node_id, target_id, "CROSS_REFERENCE")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=node.node_id,
                                target_id=target_id,
                                relation_type=RelationType.REFERENCES,
                                strength=0.9,
                                evidence=f"Cross-reference: '{ref_text}' in node {node.node_id[:8]}",
                                projections=["citation"],
                                metadata={"ref_text": ref_text, "strategy": "regex_xref"},
                            ).to_knowledge_edge()
                        )

        # 2. CITES edges (bibliography markers)
        for node in nodes:
            for match in _RE_BIBLIO_MARKER.finditer(node.content):
                marker = match.group(1).strip()
                # Try to resolve to a bibliography section node
                target_id = self._resolve_bibliography(marker, nodes, doc_id)
                if target_id:
                    key = (node.node_id, target_id, "CITES")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=node.node_id,
                                target_id=target_id,
                                relation_type=RelationType.REFERENCES,
                                strength=0.8,
                                evidence=f"Bibliography reference [{marker}] in node {node.node_id[:8]}",
                                projections=["citation"],
                                metadata={"bib_marker": marker, "strategy": "regex_biblio"},
                            ).to_knowledge_edge()
                        )

            for match in _RE_PAREN_CITE.finditer(node.content):
                author = match.group(1).strip()
                year = match.group(2).strip()
                ref_text = f"{author}, {year}"
                target_id = self._resolve_internal_ref(ref_text, headings)
                if target_id:
                    key = (node.node_id, target_id, "CITES")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=node.node_id,
                                target_id=target_id,
                                relation_type=RelationType.REFERENCES,
                                strength=0.7,
                                evidence=f"Citation '{ref_text}' in node {node.node_id[:8]}",
                                projections=["citation"],
                                metadata={"ref_text": ref_text, "strategy": "regex_paren_cite"},
                            ).to_knowledge_edge()
                        )

        # 3. REGULATES edges
        for node in nodes:
            for match in _RE_REGULATION.finditer(node.content):
                reg = match.group(0).upper().strip()
                reg_id = f"entity:{reg.lower()}"
                key = (reg_id, node.node_id, "REGULATES")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=reg_id,
                            target_id=node.node_id,
                            relation_type=RelationType.REFERENCES,
                            strength=1.0,
                            evidence=f"Regulation '{reg}' referenced in node {node.node_id[:8]}",
                            projections=["citation"],
                            metadata={"regulation": reg, "strategy": "regex_regulation"},
                        ).to_knowledge_edge()
                    )

        logger.info(
            "CitationDAG: %d edges from doc %s",
            len(all_edges),
            doc_id[:8],
        )
        return all_edges

    def _resolve_internal_ref(
        self,
        ref_text: str,
        headings: dict[str, list[ASTNode]],
    ) -> str | None:
        """Resolve a cross-reference to a heading node ID.

        Strips prefixes (``§``, ``Section``, etc.) and does case-insensitive
        fuzzy matching against heading titles.
        """
        # Normalise: remove "§", "Section", etc.
        clean = ref_text
        for prefix in ["§", "Section", "Sec.", "Chapter", "Ch.", "Appendix",
                        "Fig.", "Table", "Figure"]:
            if clean.lower().startswith(prefix.lower()):
                clean = clean[len(prefix):].strip()
                break

        clean = clean.lower().strip()

        # Try exact match first, then prefix match
        for norm_title, target_nodes in headings.items():
            if clean == norm_title or norm_title.startswith(clean) or clean.startswith(norm_title):
                return target_nodes[0].node_id

        return None

    def _resolve_bibliography(
        self,
        marker: str,  # noqa: ARG002
        nodes: list[ASTNode],
        doc_id: str,  # noqa: ARG002
    ) -> str | None:
        """Resolve a bibliography marker to the References section heading node.

        Looks for a heading named 'References', 'Bibliography', 'Works Cited',
        or 'Further Reading'. Returns the section node ID.

        Note:
            Currently resolves to the section heading, not the specific entry.
            Full resolution to individual bibliography items is planned for a
            future enhancement.
        """
        ref_section_id: str | None = None
        ref_section_children: set[str] = set()

        for node in nodes:
            title_lower = node.content.lower().strip()
            if title_lower in ("references", "bibliography", "works cited", "further reading"):
                ref_section_id = node.node_id
                # Collect child node IDs
                for child_ref in node.children:
                    if isinstance(child_ref, str):
                        ref_section_children.add(child_ref)
                break

        if not ref_section_id:
            return ref_section_id

        return ref_section_id
