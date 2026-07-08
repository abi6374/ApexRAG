"""
graph/dags/policy_dag.py — PolicyDAG Builder.

Creates policy-related edges from document content and system rules with
``projection=["policy"]``.

Strategies (deterministic, no LLM calls):
    1. **GOVERNS** — Policy/regulation → AST node it governs
    2. **REFERENCES** — Policy/regulation → other regulation
    3. **DEPENDS_ON** — Policy → policy dependency
    4. **DEFINES** — Document section → policy it defines

Usage:
    builder = PolicyDagBuilder(storage)
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
from apex_rag.models.unified_models import ASTNode, KnowledgeEdge, NodeType

logger = logging.getLogger("apex_rag.graph.dags.policy_dag")

# ── Policy regex patterns ─────────────────────────────────────────────────

# Policy statements: "X shall/must/will Y"
_RE_POLICY = re.compile(
    r"(?P<subject>[A-Z][A-Za-z\s]{2,50}?)"
    r"\s+(shall|must|will|should|may|is\s+required\s+to)\s+"
    r"(?P<condition>.{5,200}?)(?:\.|;|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Regulation references (shared with EntityDAG and CitationDAG)
_RE_REGULATION = re.compile(
    r"\b(GDPR|HIPAA|SOX|PCI(?:\sDSS)?|ISO\s*\d+|FASB|IFRS|GAAP|"
    r"ESG|KYC|AML|CCPA|ADA|OSHA|EPA)\b",
    re.IGNORECASE,
)


class PolicyDagBuilder:
    """Builds PolicyDAG edges from AST nodes and system policy rules.

    Both document-level policies (extracted via regex) and system-level
    policies (from PolicyEngine/RoleProfile) are supported.
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
        """Build PolicyDAG edges for a document from its AST nodes.

        Args:
            nodes:     AST nodes for the document.
            doc_id:    The document ID.
            tenant_id: Tenant isolation boundary (reserved).

        Returns:
            A list of :class:`KnowledgeEdge` objects with projection=["policy"].
        """
        all_edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()
        policy_id_counter: int = 0

        def _policy_id() -> str:
            nonlocal policy_id_counter
            policy_id_counter += 1
            return f"policy:{doc_id[:8]}_{policy_id_counter}"

        # 1. Extract policy statements and create GOVERNS edges
        for node in nodes:
            for match in _RE_POLICY.finditer(node.content):
                subject = match.group("subject").strip()
                condition = match.group("condition").strip()
                pid = _policy_id()

                # GOVERNS: policy → node
                key_gov = (pid, node.node_id, "GOVERNS")
                if key_gov not in seen:
                    seen.add(key_gov)
                    all_edges.append(
                        GraphEdge(
                            source_id=pid,
                            target_id=node.node_id,
                            relation_type=RelationType.REFERENCES,
                            strength=0.8,
                            evidence=f"Policy '{subject}' governs node {node.node_id[:8]}",
                            projections=["policy"],
                            metadata={
                                "policy_subject": subject,
                                "policy_condition": condition,
                                "source": "document_extraction",
                            },
                        ).to_knowledge_edge()
                    )

                # DEFINES: node → policy (if this is a heading section)
                if node.node_type == NodeType.HEADING:
                    key_def = (node.node_id, pid, "DEFINES")
                    if key_def not in seen:
                        seen.add(key_def)
                        all_edges.append(
                            GraphEdge(
                                source_id=node.node_id,
                                target_id=pid,
                                relation_type=RelationType.SUPPORTS,
                                strength=0.9,
                                evidence=f"Section '{node.content}' defines policy '{subject}'",
                                projections=["policy"],
                                metadata={"section_title": node.content, "policy_subject": subject},
                            ).to_knowledge_edge()
                        )

        # 2. Regulation references
        for node in nodes:
            for match in _RE_REGULATION.finditer(node.content):
                reg = match.group(0).upper().strip()
                reg_id = f"regulation:{reg.lower()}"

                # GOVERNS: regulation → node
                key = (reg_id, node.node_id, "GOVERNS")
                if key not in seen:
                    seen.add(key)
                    all_edges.append(
                        GraphEdge(
                            source_id=reg_id,
                            target_id=node.node_id,
                            relation_type=RelationType.REFERENCES,
                            strength=1.0,
                            evidence=f"Regulation '{reg}' governs node {node.node_id[:8]}",
                            projections=["policy"],
                            metadata={"regulation": reg, "source": "document_extraction"},
                        ).to_knowledge_edge()
                    )

        logger.info(
            "PolicyDAG: %d edges from doc %s",
            len(all_edges),
            doc_id[:8],
        )
        return all_edges

    async def build_from_system_rules(
        self,
        rules: list[Any],
        role_profiles: list[Any] | None = None,
        *,
        tenant_id: str = "default",  # noqa: ARG002
    ) -> list[KnowledgeEdge]:
        """Build PolicyDAG edges from system-level policy rules.

        Args:
            rules:         List of CustomRuleRow objects.
            role_profiles: Optional list of RoleProfileRow objects.
            tenant_id:     Tenant isolation boundary.

        Returns:
            PolicyDAG edges from system policies.
        """
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()

        for rule in rules:
            name = getattr(rule, "name", "")
            rule_type = getattr(rule, "rule_type", "")
            if not name:
                continue

            rule_id = f"policy:system:{name.lower().replace(' ', '_')}"

            # References: system rule → the rule type / resource
            resource = f"resource:{rule_type}"
            key = (rule_id, resource, "REFERENCES")
            if key not in seen:
                seen.add(key)
                edges.append(
                    GraphEdge(
                        source_id=rule_id,
                        target_id=resource,
                        relation_type=RelationType.REFERENCES,
                        strength=1.0,
                        evidence=f"System policy '{name}' references '{rule_type}'",
                        projections=["policy"],
                        metadata={"rule_name": name, "rule_type": rule_type, "source": "system"},
                    ).to_knowledge_edge()
                )

        # Role profile → GOVERNS edges
        if role_profiles:
            for rp in role_profiles:
                role_name = getattr(rp, "name", "") or getattr(rp, "role_name", "")
                if not role_name:
                    continue
                role_id = f"policy:role:{role_name.lower().replace(' ', '_')}"
                resource = "resource:document"
                key = (role_id, resource, "GOVERNS")
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        GraphEdge(
                            source_id=role_id,
                            target_id=resource,
                            relation_type=RelationType.REFERENCES,
                            strength=0.9,
                            evidence=f"Role '{role_name}' governs document access",
                            projections=["policy"],
                            metadata={"role": role_name, "source": "system"},
                        ).to_knowledge_edge()
                    )

        logger.info("PolicyDAG (system): %d edges", len(edges))
        return edges
