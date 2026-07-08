"""
agents/planner/role.py — Role Planner Agent.

Applies role-based constraints to the plan **before** navigation runs.
This is a **deterministic** stage — no LLM calls.

Role profiles are loaded from the database first (via :class:`RoleManager`).
If no profile exists in the DB, the 7 built-in hardcoded configs are used
as fallback. This ensures backward compatibility while enabling unlimited
custom roles stored as database objects.

Two enforcement mechanisms:
    1. **Hard filters**: ``visible_node_types`` / ``hidden_node_types``
       — nodes of hidden types are denied before navigation.
    2. **Soft preferences**: ``ranking_weights`` and
       ``retrieval_preferences`` — guide the navigator's priority.

Usage:
    role_planner = RolePlannerAgent(storage=apex_storage)
    plan = await role_planner.process(plan, context)
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from apex_rag.agents.planner.models import EnrichedPlan, PlanningContext

logger = logging.getLogger("apex_rag.agents.planner.role")

# ── Built-in fallback configs (used when no DB profile exists) ─────────


ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "SuperAdmin": {
        "retrieval_preferences": {"mode": "exhaustive", "freshness_weight": 1.0},
        "ranking_weights": {"vector": 0.3, "keyword": 0.3, "structural": 0.4},
        "hidden_node_types": [],
    },
    "TenantAdmin": {
        "retrieval_preferences": {"mode": "exhaustive", "freshness_weight": 0.9},
        "ranking_weights": {"vector": 0.3, "keyword": 0.3, "structural": 0.4},
        "hidden_node_types": [],
    },
    "Manager": {
        "retrieval_preferences": {"mode": "balanced", "freshness_weight": 0.8},
        "ranking_weights": {"vector": 0.3, "keyword": 0.3, "structural": 0.4},
        "hidden_node_types": [],
    },
    "Analyst": {
        "retrieval_preferences": {"mode": "balanced", "freshness_weight": 0.7},
        "ranking_weights": {"vector": 0.25, "keyword": 0.35, "structural": 0.4},
        "hidden_node_types": [],
    },
    "Auditor": {
        "retrieval_preferences": {"mode": "exhaustive", "freshness_weight": 0.6, "audit_mode": True},
        "ranking_weights": {"vector": 0.2, "keyword": 0.3, "structural": 0.5},
        "hidden_node_types": [],
    },
    "Viewer": {
        "retrieval_preferences": {"mode": "strict", "freshness_weight": 0.5},
        "ranking_weights": {"vector": 0.2, "keyword": 0.4, "structural": 0.4},
        "hidden_node_types": [],
    },
    "Guest": {
        "retrieval_preferences": {"mode": "strict", "freshness_weight": 0.3},
        "ranking_weights": {"vector": 0.1, "keyword": 0.5, "structural": 0.4},
        "hidden_node_types": ["IMAGE"],
    },
}


class RolePlannerAgent:
    """Applies role-based constraints to the plan deterministically.

    Loads RoleProfile objects from the database when ``storage`` is provided.
    Falls back to hardcoded ``ROLE_CONFIGS`` when no DB profile is found
    or when ``storage`` is ``None``.

    Two enforcement mechanisms:
        1. **Hard filters**: ``visible_node_types`` / ``hidden_node_types``
        2. **Soft preferences**: ``ranking_weights`` and ``retrieval_preferences``
    """

    def __init__(self, storage: Any | None = None) -> None:
        self._storage = storage
        self._role_configs = ROLE_CONFIGS
        self._custom_configs: dict[str, dict[str, Any]] = {}
        self._db_cache: dict[str, dict[str, Any]] = {}  # role_name → config dict

    async def process(self, plan: EnrichedPlan, context: PlanningContext) -> EnrichedPlan:
        """Apply role-based constraints to the plan.

        Loads role profiles from the database if storage is available.
        Falls back to hardcoded or registered configs.

        Args:
            plan:    The plan from KnowledgePlannerAgent.
            context: Runtime context with tenant information.

        Returns:
            The enriched plan with role-based constraints applied.
        """
        tenant = context.tenant_context
        if tenant is None:
            return self._apply_role_config(plan, "Viewer")

        merged_config: dict[str, Any] = {}
        for role in tenant.roles:
            config = await self._resolve_config(role, tenant.tenant_id)
            self._merge_configs(merged_config, config)

        # Only apply fields that were explicitly set (not default empty dicts)
        if merged_config.get("retrieval_preferences"):
            plan.retrieval_preferences = merged_config["retrieval_preferences"]
        if merged_config.get("ranking_weights"):
            plan.ranking_weights = merged_config["ranking_weights"]
        if merged_config.get("hidden_node_types"):
            plan.hidden_node_types = merged_config["hidden_node_types"]
        if merged_config.get("visible_node_types") is not None:
            plan.visible_node_types = merged_config["visible_node_types"]
        if merged_config.get("applied_policies"):
            plan.applied_policies = merged_config["applied_policies"]

        return plan

    async def _resolve_config(
        self, role_name: str, tenant_id: str
    ) -> dict[str, Any]:
        """Resolve a role's config, checking DB → custom → hardcoded.

        Args:
            role_name: The role name (e.g. 'ComplianceOfficer').
            tenant_id: The tenant ID.

        Returns:
            A config dict, or empty dict if no config found.
        """
        # 1. Check DB cache first
        cache_key = f"{tenant_id}:{role_name}"
        if cache_key in self._db_cache:
            return self._db_cache[cache_key]

        # 2. Try database
        if self._storage is not None:
            try:
                row = await self._storage.get_role_profile(role_name, tenant_id)
                if row is not None:
                    config = self._row_to_config(row)
                    self._db_cache[cache_key] = config
                    return config
            except Exception as exc:
                logger.warning(
                    "Failed to load role profile '%s' from DB: %s", role_name, exc
                )

        # 3. Check custom configs (registered via register_role_config)
        if role_name in self._custom_configs:
            return self._custom_configs[role_name]

        # 4. Fall back to hardcoded configs
        return self._role_configs.get(role_name, {})

    def register_role_config(self, role_name: str, config: dict[str, Any]) -> None:
        """Register or override a role configuration at runtime.

        Args:
            role_name: The role name (e.g. 'ComplianceOfficer').
            config:    Dict with keys: retrieval_preferences, ranking_weights,
                       hidden_node_types, visible_node_types.
        """
        self._custom_configs[role_name] = config

    def _apply_role_config(self, plan: EnrichedPlan, role: str) -> EnrichedPlan:
        """Apply a single role's config to the plan."""
        config = self._role_configs.get(role, {})
        if not config:
            return plan

        plan.retrieval_preferences = config.get("retrieval_preferences", {})
        plan.ranking_weights = config.get("ranking_weights", {})
        plan.hidden_node_types = config.get("hidden_node_types", [])
        plan.visible_node_types = config.get("visible_node_types", None)
        plan.applied_policies = [f"builtin:{role}"]
        return plan

    def _row_to_config(self, row: Any) -> dict[str, Any]:
        """Convert a RoleProfileRow to a config dict.

        Args:
            row: A database RoleProfileRow instance.

        Returns:
            A config dict compatible with ROLE_CONFIGS format.
        """
        config: dict[str, Any] = {}

        if row.ranking_weights:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                config["ranking_weights"] = json.loads(row.ranking_weights)

        if row.visible_node_types:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                visible = json.loads(row.visible_node_types)
                if isinstance(visible, list):
                    config["visible_node_types"] = visible

        if row.hidden_node_types:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                hidden = json.loads(row.hidden_node_types)
                if isinstance(hidden, list):
                    config["hidden_node_types"] = hidden

        if row.retrieval_preferences:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                config["retrieval_preferences"] = json.loads(row.retrieval_preferences)

        if row.field_visibility:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                config["field_visibility"] = json.loads(row.field_visibility)

        config["applied_policies"] = [f"role_profile:{row.name}"]
        return config

    @staticmethod
    def _merge_configs(base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Merge overlay config into base config (overlay takes precedence)."""
        for key in ("retrieval_preferences", "ranking_weights"):
            if key in overlay:
                existing = base.setdefault(key, {})
                existing.update(overlay[key])

        for key in ("hidden_node_types", "applied_policies"):
            if key in overlay:
                existing = base.setdefault(key, [])
                for item in overlay[key]:
                    if item not in existing:
                        existing.append(item)

        if "visible_node_types" in overlay:
            base["visible_node_types"] = overlay["visible_node_types"]

        if "field_visibility" in overlay:
            existing_fv = base.setdefault("field_visibility", {})
            existing_fv.update(overlay["field_visibility"])
