"""
graph/dags/entity_dag.py — EntityDAG Builder.

Extracts entities from AST nodes and TemporalFacts, then creates
typed KnowledgeEdges with ``projection=["entity"]``.

Strategies (deterministic, no LLM calls):
    1. MENTIONED_IN — Entity → AST node where it appears
    2. RELATED_TO   — Entity → Entity (co-occurrence in same section)
    3. EMPLOYS      — Organization → Person
    4. PRODUCES     — Organization → Product

Usage:
    builder = EntityDagBuilder(storage)
    edges = await builder.build(nodes, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import ASTNode, KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.entity_dag")

# ── Entity regex patterns ─────────────────────────────────────────────────

_RE_ORGANIZATION = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s"
    r"(Corporation|Corp|Inc|LLC|Ltd|PLC|GmbH|SA|NV|AG|Co|Group)\b"
)

_RE_PERSON = re.compile(
    r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)?\s*([A-Z][a-z]+)\s([A-Z][a-z]{1,20})\b"
)

_RE_REGULATION = re.compile(
    r"\b(GDPR|HIPAA|SOX|PCI(?:\sDSS)?|ISO\s*\d+|FASB|IFRS|GAAP|"
    r"ESG|KYC|AML|CCPA|ADA|OSHA|EPA)\b",
    re.IGNORECASE,
)

_RE_METRIC = re.compile(
    r"\b(Revenue|Profit|Margin|EPS|EBITDA|Net\sIncome|Operating\sIncome|"
    r"Gross\sMargin|Growth|Cost|Expense|Tax|Dividend|Yield)\b",
    re.IGNORECASE,
)


class EntityDagBuilder:
    """Builds EntityDAG edges from AST nodes and extracted facts.

    All produced edges carry ``projection=["entity"]`` and are tagged
    with entity metadata (entity_text, entity_type).
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        nodes: list[ASTNode],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build EntityDAG edges for a document from its AST nodes.

        Runs 4 strategies in order:
            1. Extract entities from node content
            2. MENTIONED_IN — entity → AST node
            3. RELATED_TO   — co-occurring entities
            4. EMPLOYS / PRODUCES — organization relationships

        Args:
            nodes:    AST nodes for the document.
            doc_id:   The document ID.
            tenant_id: Tenant isolation boundary (reserved for future use).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["entity"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        # 1. Extract entities from node content
        entity_map: dict[str, dict[str, Any]] = {}
        for node in nodes:
            entities = self._extract_entities(node.content)
            for entity_text, entity_type in entities:
                norm = entity_text.lower()
                if norm not in entity_map:
                    entity_map[norm] = {"text": entity_text, "type": entity_type, "node_ids": set()}
                entity_map[norm]["node_ids"].add(node.node_id)

        if not entity_map:
            return all_edges

        def _entity_id(text: str) -> str:
            return f"entity:{text.lower().replace(' ', '_')}"

        # 2. MENTIONED_IN edges
        for norm, info in entity_map.items():
            eid = _entity_id(norm)
            for nid in info["node_ids"]:
                key = (eid, nid, "MENTIONED_IN")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=eid,
                            target_id=nid,
                            relation_type=RelationType.REFERENCES,
                            strength=0.8,
                            evidence=f"Entity '{info['text']}' ({info['type']}) in node {nid[:8]}",
                            projections=["entity"],
                            metadata={"entity_text": info["text"], "entity_type": info["type"]},
                        ).to_knowledge_edge()
                    )

        # 3. RELATED_TO edges (co-occurrence in same nodes)
        items = list(entity_map.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                norm_a, info_a = items[i]
                norm_b, info_b = items[j]
                shared = info_a["node_ids"] & info_b["node_ids"]
                if shared:
                    eid_a = _entity_id(norm_a)
                    eid_b = _entity_id(norm_b)
                    key = (eid_a, eid_b, "RELATED_TO")
                    if key not in seen:
                        seen.add(key)
                        strength = min(0.5 + 0.1 * len(shared), 1.0)
                        all_edges.append(
                            GraphEdge(
                                source_id=eid_a,
                                target_id=eid_b,
                                relation_type=RelationType.SUPPORTS,
                                strength=strength,
                                evidence=f"Co-occurrence: '{info_a['text']}' and '{info_b['text']}' "
                                f"in {len(shared)} node(s)",
                                projections=["entity"],
                                metadata={
                                    "entity_a": info_a["text"],
                                    "entity_b": info_b["text"],
                                    "entity_type_a": info_a["type"],
                                    "entity_type_b": info_b["type"],
                                },
                            ).to_knowledge_edge()
                        )

        # 4. EMPLOYS / PRODUCES edges
        orgs = {k: v for k, v in entity_map.items() if v["type"] == "organization"}
        persons = {k: v for k, v in entity_map.items() if v["type"] == "person"}
        products = {k: v for k, v in entity_map.items() if v["type"] == "product"}

        for norm_org, org_info in orgs.items():
            eid_org = _entity_id(norm_org)
            for norm_person, person_info in persons.items():
                shared = org_info["node_ids"] & person_info["node_ids"]
                if shared:
                    eid_person = _entity_id(norm_person)
                    key = (eid_org, eid_person, "EMPLOYS")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=eid_org,
                                target_id=eid_person,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.7,
                                evidence=f"'{org_info['text']}' employs '{person_info['text']}'",
                                projections=["entity"],
                                metadata={
                                    "relation": "employs",
                                    "employer": org_info["text"],
                                    "employee": person_info["text"],
                                },
                            ).to_knowledge_edge()
                        )
            for norm_prod, prod_info in products.items():
                shared = org_info["node_ids"] & prod_info["node_ids"]
                if shared:
                    eid_prod = _entity_id(norm_prod)
                    key = (eid_org, eid_prod, "PRODUCES")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=eid_org,
                                target_id=eid_prod,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.6,
                                evidence=f"'{org_info['text']}' produces '{prod_info['text']}'",
                                projections=["entity"],
                                metadata={
                                    "relation": "produces",
                                    "producer": org_info["text"],
                                    "product": prod_info["text"],
                                },
                            ).to_knowledge_edge()
                        )

        logger.info(
            "EntityDAG: %d entities → %d edges from doc %s",
            len(entity_map),
            len(all_edges),
            doc_id[:8],
        )
        return all_edges

    async def build_from_facts(
        self,
        facts: list[Any],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build EntityDAG edges from extracted TemporalFact objects.

        Creates ``MENTIONED_IN`` edges for fact subjects.

        Args:
            facts:    List of TemporalFact objects.
            doc_id:   The document ID.
            tenant_id: Tenant isolation boundary (reserved for future use).

        Returns:
            EntityDAG edges derived from structured facts.
        """
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        for fact in facts:
            subject = getattr(fact, "subject", "")
            source_node = getattr(fact, "source_node_id", "")
            confidence = getattr(fact, "confidence", 0.7)

            if not subject or not source_node:
                continue

            eid = f"entity:{subject.lower().replace(' ', '_')}"
            key = (eid, source_node, "MENTIONED_IN")
            if key not in seen:
                seen.add(key)
                edges.append(
                    GraphEdge(
                        source_id=eid,
                        target_id=source_node,
                        relation_type=RelationType.REFERENCES,
                        strength=confidence,
                        evidence=f"Entity '{subject}' extracted as fact in node {source_node[:8]}",
                        projections=["entity"],
                        metadata={"entity_text": subject, "source": "fact_extraction"},
                    ).to_knowledge_edge()
                )

        logger.info("EntityDAG (facts): %d edges from doc %s", len(edges), doc_id[:8])
        return edges

    def _extract_entities(self, content: str) -> list[tuple[str, str]]:
        """Extract (entity_name, entity_type) pairs from text.

        Pure regex — no LLM calls. Duplicates are skipped.
        """
        entities: list[tuple[str, str]] = []
        seen: set[str] = set()

        if not content:
            return entities

        for match in _RE_ORGANIZATION.finditer(content):
            name = match.group(0).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append((name, "organization"))

        for match in _RE_PERSON.finditer(content):
            name = f"{match.group(1)} {match.group(2)}"
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append((name, "person"))

        for match in _RE_REGULATION.finditer(content):
            name = match.group(0).upper().strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                entities.append((name, "regulation"))

        for match in _RE_METRIC.finditer(content):
            name = match.group(0).strip()
            if name.lower() not in seen and len(name) > 2:
                seen.add(name.lower())
                entities.append((name, "metric"))

        return entities
