"""
agents/planner/dag_router.py — Query-Need Classifier for Lazy DAGs.

Determines which lazy Knowledge DAGs are needed for a given query
using fast heuristic/keyword matching (<10ms, no LLM call).

Extension point: swap in a learned/RL-trained router later without
changing the caller interface.

Usage:
    router = DAGRouter()
    needed = router.classify("What entities are mentioned?")
    # → {"entity"}
"""

from __future__ import annotations

import re
from typing import Any

# ── Keyword signals per DAG projection ────────────────────────────────────

_ENTITY_SIGNALS: list[str] = [
    r"\bwho\b", r"\bwhom\b", r"\bwhose\b",
    r"\bperson\b", r"\bpeople\b", r"\bemployee",
    r"\borganization", r"\bcompany", r"\bcorporation",
    r"\bentity", r"\bentities",
    r"\bmention", r"\brefers?\s+to",
    r"\bindividual", r"\bstakeholder",
    r"\bfounder", r"\bceo\b", r"\bexecutive",
    r"\bpartner(ship)?\b",
    r"\bemploy(s|er|ee|ing|ment)?\b",
    r"\bsubject\b", r"\bparty\b", r"\bparties\b",
]

_CITATION_SIGNALS: list[str] = [
    r"\bcite", r"\bcitation", r"\breference",
    r"\bbibliography", r"\breferences?\s+section",
    r"\bsee\s+(§|section|chapter)",
    r"\bcf\.", r"\bvide\b",
    r"\baccording\s+to",
    r"\bas\s+mentioned\s+in",
    r"\bfootnote", r"\bendnote",
    r"\[\d+\]",  # inline citation markers like [1]
    r"\([A-Z][a-z]+.*,\s*\d{4}\)",  # (Smith, 2020)
    r"\bworks?\s+cited\b",
    r"\bfurther\s+reading\b",
]

_POLICY_SIGNALS: list[str] = [
    r"\bpolic(y|ies)\b",
    r"\bregulat", r"\bregulation",
    r"\bgovern", r"\bgovernance",
    r"\bshall\b", r"\bmust\b", r"\bis\s+required\b",
    r"\bcompliance", r"\bcompliant",
    r"\bmandatory", r"\bobligation",
    r"\bstandard", r"\bprocedure",
    r"\bprotocol", r"\bguideline",
    r"\bpermission", r"\bauthoriz",
    r"\brestrict", r"\bprohibit",
    r"\brule(s)?\b", r"\blaw(s)?\b", r"\blegal\b",
    r"\bGDPR", r"\bHIPAA", r"\bSOX\b", r"\bESG\b",
    r"\bCCPA", r"\bADA\b", r"\bOSHA\b", r"\bEPA\b",
]


class DAGRouter:
    """Fast heuristic classifier that maps a query to needed DAG projections.

    Uses keyword/regex matching against the query text.  Designed to
    run in <10ms without any LLM call.

    Extension point:
        Replace ``classify()`` with a learned classifier (e.g. a small
        fine-tuned transformer) without changing the caller interface.
    """

    def __init__(self) -> None:
        self._entity_re = re.compile("|".join(_ENTITY_SIGNALS), re.IGNORECASE)
        self._citation_re = re.compile("|".join(_CITATION_SIGNALS), re.IGNORECASE)
        self._policy_re = re.compile("|".join(_POLICY_SIGNALS), re.IGNORECASE)

    def classify(self, query: str, *, planner_data: dict[str, Any] | None = None) -> frozenset[str]:
        """Determine which lazy DAG projections are needed for a query.

        Uses heuristic regex matching on the query text (<10ms, no LLM
        call).  Optionally also inspects the ``QueryPlannerAgent``'s
        output (``planner_data``) for additional signals like
        ``query_type`` and ``entity_hints``.

        The regex-based step 1 does all keyword matching.  Planner data
        (step 2) augments decisions based on query type classification.

        Args:
            query:        The user's natural-language query.
            planner_data: Optional output from ``QueryPlannerAgent.plan_query()``
                          containing ``query_type`` and ``entity_hints``.

        Returns:
            A frozenset of DAG projection tags needed (e.g. ``{\"entity\"}``).
        """
        needed: set[str] = set()

        # ── Step 1: Regex keyword matching on raw query ───────────────
        if self._entity_re.search(query):
            needed.add("entity")
        if self._citation_re.search(query):
            needed.add("citation")
        if self._policy_re.search(query):
            needed.add("policy")

        # ── Step 2: Planner data signals (if available) ───────────────
        self._apply_planner_data(needed, planner_data)

        return frozenset(needed)

    def classify_from_plan(
        self,
        plan: Any,
        query: str,
    ) -> frozenset[str]:
        """Convenience method: classify from an ``EnrichedPlan`` object.

        Args:
            plan:  The :class:`EnrichedPlan` from the planning pipeline.
            query: The original user query.

        Returns:
            A frozenset of needed DAG projection tags.
        """
        planner_data = None
        if hasattr(plan, "query_type") and hasattr(plan, "entity_hints"):
            planner_data = {
                "query_type": plan.query_type,
                "entity_hints": plan.entity_hints if hasattr(plan, "entity_hints") else {},
            }
        elif hasattr(plan, "to_dict"):
            planner_data = plan.to_dict()

        return self.classify(query, planner_data=planner_data)

    @staticmethod
    def _apply_planner_data(
        needed: set[str],
        planner_data: dict[str, Any] | None,
    ) -> None:
        """Enrich the ``needed`` set with signals from planner data.

        Shared between the heuristic and ML classifiers to keep
        planner-signal logic in one place.

        Args:
            needed:       Mutable set of DAG projection tags to update.
            planner_data: Optional dict with ``query_type`` and
                          ``entity_hints`` keys, or ``None``.
        """
        if not planner_data:
            return
        qtype = planner_data.get("query_type", "")
        if qtype in ("LEGAL", "FINANCIAL", "COMPLIANCE"):
            needed.add("policy")
        if qtype in ("MULTI_DOCUMENT", "COMPARATIVE"):
            needed.add("citation")
        entity_hints = planner_data.get("entity_hints", {})
        if entity_hints:
            needed.add("entity")
