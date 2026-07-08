"""
agents/planner/knowledge.py — Knowledge Planner Agent.

Validates and enriches the entity hints, structural domain, and expected
node types produced by the QueryPlannerAgent.

This stage is **LLM-free** — it uses deterministic keyword analysis and
domain mapping tables to validate and fill gaps in the plan.

Domain map:
    financial  → TABLE, PARAGRAPH
    legal      → PARAGRAPH, LIST, HEADING
    code       → CODE, PARAGRAPH
    technical  → PARAGRAPH, TABLE, LIST
    medical    → PARAGRAPH, TABLE
    general    → PARAGRAPH, LIST
"""

from __future__ import annotations

import re

from apex_rag.agents.planner.models import EnrichedPlan, PlanningContext

# ── Domain → expected node type mapping ─────────────────────────────────


DOMAIN_NODE_TYPES: dict[str, list[str]] = {
    "financial": ["TABLE", "PARAGRAPH", "LIST"],
    "legal": ["PARAGRAPH", "LIST", "HEADING"],
    "code": ["CODE", "PARAGRAPH"],
    "technical": ["PARAGRAPH", "TABLE", "LIST"],
    "medical": ["PARAGRAPH", "TABLE"],
    "scientific": ["PARAGRAPH", "TABLE", "LIST"],
    "educational": ["PARAGRAPH", "LIST", "HEADING"],
    "general": ["PARAGRAPH", "LIST"],
}

# ── Common financial/legal entities for fallback extraction ──────────────

COMMON_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "ticker": re.compile(r"\b[A-Z]{1,5}\b"),  # Stock tickers
    "year": re.compile(r"\b(19|20)\d{2}\b"),  # Years
    "currency": re.compile(r"\$[\d,]+(?:\.\d+)?"),  # Dollar amounts
    "percentage": re.compile(r"\d+\.?\d*\%"),  # Percentages
    "quarter": re.compile(r"\bQ[1-4]\b", re.IGNORECASE),  # Quarters
}


class KnowledgePlannerAgent:
    """Validates and enriches the plan with domain knowledge.

    This agent runs **after** QueryPlannerAgent and enriches the plan with:
        - Validated entity hints (filters out noise from LLM output)
        - Structural domain mapping (domain → expected node types)
        - Fallback entity extraction via keyword patterns if LLM returned none

    No additional LLM call is made — all logic is deterministic.
    """

    def __init__(self) -> None:
        self._domain_node_types = DOMAIN_NODE_TYPES

    async def process(self, plan: EnrichedPlan, context: PlanningContext) -> EnrichedPlan:
        """Enrich the plan with knowledge-layer data.

        Args:
            plan:    The plan from QueryPlannerAgent.
            context: Runtime context (doc_id, domain, tenant).

        Returns:
            The enriched plan with validated entity hints and domain data.
        """
        _ = context  # Reserved for future context-dependent enrichment
        # 1. Validate structural domain
        domain = plan.structural_domain
        if domain and domain.lower() in self._domain_node_types:
            # Merge expected_node_types with domain defaults
            known_types = self._domain_node_types[domain.lower()]
            plan.expected_node_types = list(
                dict.fromkeys(known_types + plan.expected_node_types)
            )  # Deduplicate preserving order

        # 2. Fallback entity extraction for sub-queries with no hints
        for sq in plan.sub_queries:
            if sq not in plan.entity_hints or not plan.entity_hints[sq]:
                plan.entity_hints[sq] = self._extract_entities(sq)

        # 3. Filter entity hints to remove empty entries
        plan.entity_hints = {
            sq: entities for sq, entities in plan.entity_hints.items() if entities
        }

        return plan

    def _extract_entities(self, text: str) -> list[str]:
        """Fallback entity extraction using keyword patterns.

        Args:
            text: A sub-query string.

        Returns:
            A list of entity strings found in the text.
        """
        entities: list[str] = []
        seen: set[str] = set()

        for _pattern_name, pattern in COMMON_ENTITY_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                match_upper = match.upper()
                if match_upper not in seen:
                    entities.append(match)
                    seen.add(match_upper)

        return entities

    def register_domain_mapping(self, domain: str, node_types: list[str]) -> None:
        """Register a custom domain → node type mapping.

        Args:
            domain:     Domain name (e.g. 'regulatory').
            node_types: List of node types (e.g. ['TABLE', 'PARAGRAPH']).
        """
        self._domain_node_types[domain.lower()] = node_types
