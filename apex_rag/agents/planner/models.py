"""
agents/planner/models.py — Data models for the multi-stage planning pipeline.

Defines:
    - :class:`EnrichedPlan`: The unified plan object passed through all planner stages.
    - :class:`PlannerStage`: Protocol that every planner stage must implement.
    - :class:`PlanningContext`: Runtime context passed to planner stages.

The planning pipeline is:
    QueryPlannerAgent → KnowledgePlannerAgent → RolePlannerAgent → TemporalPlannerAgent

Each stage enriches the EnrichedPlan with additional constraints and metadata.
The final EnrichedPlan is consumed by the Navigator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from apex_rag.enterprise.auth.models import TenantContext


@dataclass
class EnrichedPlan:
    """The unified plan object enriched by all planner stages.

    Starts empty and gains data as it passes through the pipeline:

    1. **QueryPlannerAgent** — fills ``query_type``, ``sub_queries``, ``reasoning``,
       ``entity_hints``, ``structural_domain``, ``expected_node_types``.
    2. **KnowledgePlannerAgent** — validates entity hints, enriches structural domain,
       may add keyword fallback entities.
    3. **RolePlannerAgent** — fills ``retrieval_preferences``, ``ranking_weights``,
       ``visible_node_types``, ``hidden_node_types``, ``applied_policies``.
    4. **TemporalPlannerAgent** — fills ``temporal_mode``, ``date_constraints``,
       ``version_filters``.

    The final plan is passed to :class:`~apex_rag.retrieval.agentic.navigator.ASTNavigationAgent`
    which uses the constraints to guide its tree walk.
    """

    # ── From QueryPlannerAgent ──────────────────────────────────────────
    query_type: str = "FACTUAL"
    sub_queries: list[str] = field(default_factory=list)
    reasoning: str = ""

    # ── From KnowledgePlannerAgent ──────────────────────────────────────
    entity_hints: dict[str, list[str]] = field(default_factory=dict)
    """Maps each sub-query to a list of entity names it should look for."""

    structural_domain: str | None = None
    """e.g. 'financial', 'legal', 'technical', 'code', 'medical'"""

    expected_node_types: list[str] = field(default_factory=list)
    """e.g. ['TABLE', 'PARAGRAPH'] — hints for the navigator."""

    # ── From RolePlannerAgent ───────────────────────────────────────────
    retrieval_preferences: dict[str, Any] = field(default_factory=dict)
    """e.g. {'mode': 'strict', 'freshness_weight': 0.8}"""

    ranking_weights: dict[str, float] = field(default_factory=dict)
    """e.g. {'vector': 0.0, 'keyword': 0.4, 'structural': 0.6}"""

    visible_node_types: list[str] | None = None
    """If set, only these node types are allowed (None = all types visible)."""

    hidden_node_types: list[str] = field(default_factory=list)
    """Node types that are blocked (denied nodes raise AccessDenied)."""

    applied_policies: list[str] = field(default_factory=list)
    """Names of policies that were applied during role planning."""

    # ── From TemporalPlannerAgent ───────────────────────────────────────
    temporal_mode: str | None = None
    """One of 'AS_OF', 'BETWEEN', 'TREND', 'LATEST', or None."""

    date_constraints: list[datetime] = field(default_factory=list)
    """Date bounds for the query (length 1 for AS_OF, 2 for BETWEEN)."""

    version_filters: dict[str, Any] = field(default_factory=dict)
    """Version resolution parameters, e.g. {'as_of_version': '2.0'}."""

    has_temporal_query: bool = False
    """True when the query has a temporal component (time travel)."""

    # ── Factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_planner_data(cls, data: dict[str, Any]) -> EnrichedPlan:
        """Create an EnrichedPlan from the QueryPlannerAgent's output dict.

        Args:
            data: The dict returned by ``QueryPlannerAgent.plan_query()``,
                  expected to have keys ``query_type``, ``sub_queries``,
                  ``reasoning``, ``entity_hints``, ``structural_domain``,
                  ``expected_node_types``.

        Returns:
            A new EnrichedPlan with the available data.
        """
        return cls(
            query_type=data.get("query_type", "FACTUAL"),
            sub_queries=data.get("sub_queries", []),
            reasoning=data.get("reasoning", ""),
            entity_hints=data.get("entity_hints", {}),
            structural_domain=data.get("structural_domain"),
            expected_node_types=data.get("expected_node_types", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a dict for tracing and observability."""
        return {
            "query_type": self.query_type,
            "sub_queries_count": len(self.sub_queries),
            "structural_domain": self.structural_domain,
            "temporal_mode": self.temporal_mode,
            "has_temporal_query": self.has_temporal_query,
            "visible_node_types": self.visible_node_types,
            "applied_policies": self.applied_policies,
            "entity_hints_keys": list(self.entity_hints.keys()),
        }


@dataclass
class PlanningContext:
    """Runtime context passed through the planning pipeline.

    Contains everything a planner stage might need beyond the plan itself.
    """

    tenant_context: TenantContext | None = None
    doc_id: str = ""
    domain: str = "general"


class PlannerStage(Protocol):
    """Protocol that every planner stage must implement.

    Each stage receives the current plan and context, and returns
    an enriched plan.  Stages can be composed arbitrarily.
    """

    async def process(self, plan: EnrichedPlan, context: PlanningContext) -> EnrichedPlan:
        """Enrich the plan with stage-specific data.

        Args:
            plan:    The plan from the previous stage.
            context: Runtime context (tenant, doc_id, domain).

        Returns:
            The enriched plan.
        """
        ...
