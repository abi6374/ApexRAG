"""
graph/dags/fact_dag.py — FactDAG Builder.

Creates fact relationship edges from extracted TemporalFact objects
with ``projection=["fact"]``.

Strategies (deterministic, no LLM calls):
    1. **SUPPORTS** — Two facts with same subject and compatible values
    2. **CONTRADICTS** — Two facts with same subject but conflicting values
    3. **DEPENDS_ON** — Fact → node it was extracted from
    4. **SAME_TOPIC** — Facts with the same subject from different sources

Usage:
    builder = FactDagBuilder(storage)
    edges = await builder.build(facts, doc_id="doc-123")
    for edge in edges:
        await storage.save_knowledge_edge(edge)
"""

from __future__ import annotations

import logging
from typing import Any

from apex_rag.graph.edges.models import GraphEdge, RelationType
from apex_rag.ingestion.apex_storage import ApexStorage
from apex_rag.models.unified_models import KnowledgeEdge

logger = logging.getLogger("apex_rag.graph.dags.fact_dag")


class FactDagBuilder:
    """Builds FactDAG edges from extracted TemporalFact objects.

    All edges carry ``projection=["fact"]`` with fact metadata
    including subject, predicate, confidence, and fact_type.
    """

    def __init__(self, storage: ApexStorage | None = None) -> None:
        self._storage = storage

    async def build(
        self,
        facts: list[Any],
        *,
        doc_id: str,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build FactDAG edges from TemporalFact objects.

        Runs 3 strategies:
            1. DEPENDS_ON — fact → source AST node
            2. SUPPORTS / CONTRADICTS — fact-to-fact comparison by subject
            3. SAME_TOPIC — facts with same subject from different sections

        Args:
            facts:     List of TemporalFact objects for the document.
            doc_id:    The document ID.
            tenant_id: Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["fact"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        def _fact_id(fact: Any) -> str:
            return getattr(fact, "fact_id", "")

        def _fact_subject(fact: Any) -> str:
            return getattr(fact, "subject", "") or ""

        def _fact_predicate(fact: Any) -> str:
            return getattr(fact, "predicate", "") or ""

        def _fact_object(fact: Any) -> str:
            return getattr(fact, "object", "") or ""

        def _fact_confidence(fact: Any) -> float:
            return getattr(fact, "confidence", 0.5) or 0.5

        def _fact_source_node(fact: Any) -> str:
            return getattr(fact, "source_node_id", "") or ""

        # 1. DEPENDS_ON: fact → source AST node
        for fact in facts:
            fid = _fact_id(fact)
            sid = _fact_source_node(fact)
            if fid and sid:
                key = (fid, sid, "DEPENDS_ON")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=fid,
                            target_id=sid,
                            relation_type=RelationType.DEPENDS_ON,
                            strength=_fact_confidence(fact),
                            evidence=f"Fact '{_fact_subject(fact)}' extracted from node {sid[:8]}",
                            projections=["fact"],
                            metadata={
                                "subject": _fact_subject(fact),
                                "predicate": _fact_predicate(fact),
                                "confidence": _fact_confidence(fact),
                            },
                        ).to_knowledge_edge()
                    )

        # 2. SUPPORTS / CONTRADICTS: compare facts by subject
        by_subject: dict[str, list[Any]] = {}
        for fact in facts:
            subj = _fact_subject(fact)
            if subj:
                by_subject.setdefault(subj, []).append(fact)

        for subject, subject_facts in by_subject.items():
            if len(subject_facts) < 2:
                continue

            for i in range(len(subject_facts)):
                for j in range(i + 1, len(subject_facts)):
                    a, b = subject_facts[i], subject_facts[j]
                    aid, bid = _fact_id(a), _fact_id(b)
                    obj_a, obj_b = _fact_object(a), _fact_object(b)

                    # Compare objects for contradiction detection
                    if obj_a and obj_b:
                        normalized_a = obj_a.lower().strip()
                        normalized_b = obj_b.lower().strip()

                        # Simple contradiction: different numeric values
                        is_contradiction = (
                            normalized_a != normalized_b
                            and len(normalized_a) > 2
                            and len(normalized_b) > 2
                        )

                        edge_type = (
                            RelationType.CONTRADICTS if is_contradiction else RelationType.SUPPORTS
                        )
                        key = (aid, bid, edge_type.value)
                        if key not in seen:
                            seen.add(key)
                            strength = 0.6 if is_contradiction else 0.7
                            rel_label = "contradicts" if is_contradiction else "supports"
                            all_edges.append(
                                GraphEdge(
                                    source_id=aid,
                                    target_id=bid,
                                    relation_type=edge_type,
                                    strength=strength,
                                    evidence=f"Fact '{subject}' {rel_label} fact {bid[:8]} "
                                    f"('{obj_a[:50]}' vs '{obj_b[:50]}')",
                                    projections=["fact"],
                                    metadata={
                                        "subject": subject,
                                        "object_a": obj_a,
                                        "object_b": obj_b,
                                        "relation": rel_label,
                                    },
                                ).to_knowledge_edge()
                            )

        # 3. SAME_TOPIC: facts with same subject from different source nodes
        for subject, subject_facts in by_subject.items():
            if len(subject_facts) < 2:
                continue

            source_nodes: set[str] = set()
            for fact in subject_facts:
                sid = _fact_source_node(fact)
                if sid:
                    source_nodes.add(sid)

            source_list = list(source_nodes)
            if len(source_list) < 2:
                continue

            for i in range(len(source_list)):
                for j in range(i + 1, len(source_list)):
                    key = (source_list[i], source_list[j], "SAME_TOPIC")
                    if key not in seen:
                        seen.add(key)
                        all_edges.append(
                            GraphEdge(
                                source_id=source_list[i],
                                target_id=source_list[j],
                                relation_type=RelationType.SAME_TOPIC,
                                strength=0.7,
                                evidence=f"Both nodes contain facts about '{subject}'",
                                projections=["fact"],
                                metadata={"subject": subject},
                            ).to_knowledge_edge()
                        )

        logger.info(
            "FactDAG: %d facts → %d edges from doc %s",
            len(facts),
            len(all_edges),
            doc_id[:8],
        )
        return all_edges
