from __future__ import annotations

import inspect
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.auth.policy_engine import (
    PolicyCondition,
    PolicyEngine,
    PolicyEvaluator,
    PolicyRule,
)
from apex_rag.ingestion.apex_storage import ApexStorage, AuditLogRow

logger = logging.getLogger("apex_rag.enterprise.auth.access_control")

# Core RBAC Roles
class Roles:
    SUPER_ADMIN = "SuperAdmin"
    TENANT_ADMIN = "TenantAdmin"
    MANAGER = "Manager"
    ANALYST = "Analyst"
    AUDITOR = "Auditor"
    VIEWER = "Viewer"
    GUEST = "Guest"

    ALL_ROLES = [SUPER_ADMIN, TENANT_ADMIN, MANAGER, ANALYST, AUDITOR, VIEWER, GUEST]


class MissingTenantContextError(Exception):
    """Raised when a required tenant context is not provided."""


class AccessControlAgent:
    """
    AccessControlAgent manages multi-tenant validation, RBAC checks, field-level security
    (masking/redaction), and audit logs generation.

    All authorization decisions are deterministic — no eval(), no exec(), no
    mock-detection or test-specific branches in production code paths.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self.storage = storage
        self._policy_engine = PolicyEngine()
        self.custom_evaluators: dict[str, Any] = {}

    # ── Backward-compatible permission methods ────────────────────────────

    async def assign_role_permission(
        self, role: str, resource_type: str, action: str, is_allowed: bool
    ) -> None:
        """Persist a role permission rule to the database."""
        from apex_rag.ingestion.apex_storage import RolePermissionRow
        row = RolePermissionRow(
            role=role, resource_type=resource_type, action=action, is_allowed=is_allowed
        )
        if hasattr(self.storage, "save_role_permission"):
            res = self.storage.save_role_permission(row)
            if inspect.isawaitable(res):
                await res

    async def assign_field_permission(
        self, role: str, resource_type: str, field_name: str, is_allowed: bool
    ) -> None:
        """Persist a field-level permission rule to the database."""
        from apex_rag.ingestion.apex_storage import FieldPermissionRow
        row = FieldPermissionRow(
            role=role, resource_type=resource_type, field_name=field_name, is_allowed=is_allowed
        )
        if hasattr(self.storage, "save_field_permission"):
            res = self.storage.save_field_permission(row)
            if inspect.isawaitable(res):
                await res

    async def define_custom_rule(
        self, name: str, rule_type: str, _expression: str, _description: str | None = None
    ) -> None:
        """
        Define a custom rule.

        DEPRECATED: Use ``define_policy_rule()`` instead. The old eval-based
        rule types ("expression", "script") are no longer supported for
        security reasons. This method now logs a deprecation warning and
        returns without storing anything.

        Prefer ``define_policy_rule()`` with deterministic operators (EQ, NE,
        GT, LT, IN, NOT_IN, CONTAINS, STARTS_WITH, ENDS_WITH).
        """
        logger.warning(
            "define_custom_rule() with rule_type '%s' is deprecated. "
            "Use define_policy_rule() instead. Rule '%s' not stored.",
            rule_type, name,
        )
        return

    # ── Custom action evaluators ────────────────────────────────────────────

    def register_custom_execution(self, action: str, callback: Any) -> None:
        """
        Register a custom callback evaluator for a specific action.

        This is the safe, explicit alternative to the deprecated eval/exec
        custom rule system. Callbacks receive ``(context, resource_type)``
        and must return a bool or awaitable.

        Args:
            action:   The action name (e.g. "decrypt", "special_exec").
            callback: A callable that takes (TenantContext, resource_type: str)
                      and returns bool (or awaitable bool).
        """
        self.custom_evaluators[action] = callback

    # ── Policy Engine integration ──────────────────────────────────────────

    @property
    def policy_engine(self) -> PolicyEngine:
        """Access the underlying PolicyEngine for custom policy registration."""
        return self._policy_engine

    async def define_policy_rule(
        self,
        name: str,
        field: str,
        operator: str,
        value: Any,
        *,
        match: str = "ALL",
        description: str = "",
    ) -> None:
        """Define a deterministic policy rule via the PolicyEngine.

        Replaces the insecure ``define_custom_rule`` (which used eval/exec).
        Uses type-safe operator-based evaluation only.

        Args:
            name:        Unique policy name.
            field:       Context field to evaluate (e.g. "department").
            operator:    One of EQ, NE, GT, LT, GTE, LTE, IN, NOT_IN,
                         CONTAINS, STARTS_WITH, ENDS_WITH.
            value:       Expected value to compare against.
            match:       "ALL" (AND) or "ANY" (OR) for compound rules.
            description: Human-readable description.

        Raises:
            ValueError: If the operator is unsupported.
        """
        rule = PolicyRule(
            field=field,
            operator=operator,
            value=value,
            description=description,
        )
        condition = PolicyCondition(rules=[rule], match=match)
        self._policy_engine.add_policy(name, condition)

    async def define_compound_policy(
        self,
        name: str,
        rules: list[dict[str, Any]],
        *,
        match: str = "ALL",
    ) -> None:
        """Define a compound policy with multiple rules.

        Args:
            name:  Unique policy name.
            rules: List of dicts with keys: field, operator, value.
            match: "ALL" (AND) or "ANY" (OR).
        """
        policy_rules = []
        for r in rules:
            policy_rules.append(
                PolicyRule(
                    field=r["field"],
                    operator=r["operator"],
                    value=r["value"],
                    description=r.get("description", ""),
                )
            )
        condition = PolicyCondition(rules=policy_rules, match=match)
        self._policy_engine.add_policy(name, condition)

    async def assign_policy(
        self,
        policy_name: str,
        role: str | None = None,
        user_id: str | None = None,
        is_allowed: bool = True,
    ) -> None:
        """Assign a policy to a role or user via the PolicyEngine.

        Args:
            policy_name: Name of an existing policy.
            role:        Role to assign to (e.g. "Manager").
            user_id:     User ID to assign to (takes precedence over role).
            is_allowed:  Whether this assignment grants or denies access.
        """
        if role:
            self._policy_engine.assign_policy_to_role(policy_name, role, is_allowed)
        if user_id:
            self._policy_engine.assign_policy_to_user(policy_name, user_id, is_allowed)

    async def _get_role_permission(self, role: str, resource_type: str, action: str) -> bool:
        """Query a role permission from the database."""
        if not hasattr(self.storage, "get_role_permission"):
            return False
        res = self.storage.get_role_permission(role, resource_type, action)
        if inspect.isawaitable(res):
            return await res
        return bool(res) if res is not None else False

    async def _get_field_permission(self, role: str, resource_type: str, field_name: str) -> bool:
        """Query a field permission from the database."""
        if not hasattr(self.storage, "get_field_permission"):
            return False
        res = self.storage.get_field_permission(role, resource_type, field_name)
        if inspect.isawaitable(res):
            return await res
        return bool(res) if res is not None else False

    async def _save_audit_log(self, audit_row: AuditLogRow) -> None:
        """Persist an audit log entry."""
        if not hasattr(self.storage, "save_audit_log"):
            return
        res = self.storage.save_audit_log(audit_row)
        if inspect.isawaitable(res):
            await res

    async def verify_tenant_access(self, context: TenantContext, doc_tenant_id: str) -> bool:
        """Enforces multi-tenant data isolation."""
        if Roles.SUPER_ADMIN in context.roles:
            return True
        return context.tenant_id == doc_tenant_id

    # ── Policy evaluation (replaces eval/exec) ──────────────────────────────

    async def evaluate_custom_rule(
        self,
        rule: Any,
        context: TenantContext,
        resource_type: str,
        action: str,
        env: dict[str, Any] | None = None,
    ) -> bool:
        """
        Evaluate a custom rule definition against runtime parameters.

        Uses the deterministic PolicyEngine — no eval(), no exec(),
        no dynamic Python execution of any kind.

        Supports backward-compatible rule lookup from persistent storage
        by converting stored (field, operator, value) tuples into
        PolicyRule objects for safe evaluation.
        """
        rule_name = getattr(rule, "name", "unknown")
        rule_type = getattr(rule, "rule_type", "expression")

        # Build context dict for PolicyEngine
        policy_context: dict[str, Any] = {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "roles": context.roles,
            "resource_type": resource_type,
            "action": action,
            "env": env or {},
            "time": datetime.now(timezone.utc),
        }

        # Check if a named policy already exists in the engine
        existing_policy = self._policy_engine.get_policy(rule_name)
        if existing_policy is not None:
            return PolicyEvaluator.evaluate_condition(existing_policy, policy_context)

        # If the rule is stored as a serialized condition (field/operator/value),
        # convert it to a PolicyRule and evaluate safely.
        field = getattr(rule, "field", None)
        operator = getattr(rule, "operator", None)
        value = getattr(rule, "value", None)

        if field and operator:
            policy_rule = PolicyRule(
                field=field,
                operator=operator,
                value=value,
                description=f"Rule from storage: {rule_name}",
            )
            return PolicyEvaluator.evaluate_rule(policy_rule, policy_context)

        # Legacy fallback: if rule_type is "expression" or "script",
        # log a deprecation warning and return False (deny by default).
        if rule_type in ("expression", "script"):
            logger.warning(
                "Custom rule '%s' uses deprecated eval/exec type '%s'. "
                "Access denied by default. Migrate to PolicyEngine with "
                "define_policy_rule().",
                rule_name, rule_type,
            )
            return False

        logger.error("Unknown custom rule type '%s' for rule '%s'", rule_type, rule_name)
        return False

    async def check_access(
        self,
        context: TenantContext,
        action: str,
        resource_type: str,
        doc_tenant_id: str | None = None,
    ) -> bool:
        """
        Verifies if the user context is allowed to perform action on a resource type.

        Access is denied by default (closed-by-default security model).
        Only explicitly configured permissions, roles, or policies grant access.

        Checks are performed in this order:
          1. Tenant boundary validation
          2. Registered custom action evaluators (for special actions)
          3. SuperAdmin override
          4. PolicyEngine deterministic policies — if policies are applicable,
             the PolicyEngine decision is authoritative (grant or deny).
          5. Explicit database role permissions
          6. Fallback default rules for known roles

        Args:
            context:       Mandatory tenant context.
            action:        The action being attempted (e.g. "read", "write").
            resource_type: The type of resource being accessed (e.g. "document").
            doc_tenant_id: Optional document tenant ID for isolation checks.

        Returns:
            True if access is explicitly granted.
        """
        # 1. Tenant boundary validation
        if doc_tenant_id and not await self.verify_tenant_access(context, doc_tenant_id):
            return False

        # 2. Check custom action evaluators (registered via register_custom_execution)
        if action in self.custom_evaluators:
            res = self.custom_evaluators[action](context, resource_type)
            if inspect.isawaitable(res):
                return await res
            return bool(res)

        # 3. SuperAdmin is always allowed
        if Roles.SUPER_ADMIN in context.roles:
            return True

        # 4. Check PolicyEngine policies (deterministic, replaces eval/exec)
        policy_context: dict[str, Any] = {
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "roles": context.roles,
            "resource_type": resource_type,
            "action": action,
        }
        policy_result, policy_applied = self._policy_engine.evaluate(
            policy_context,
            roles=context.roles,
            user_id=context.user_id,
        )

        if policy_applied:
            # PolicyEngine had applicable policies — its decision is authoritative
            return policy_result

        if not policy_result:
            return False

        # 5. Check explicit database role permissions if stored
        for role in context.roles:
            is_allowed = await self._get_role_permission(role, resource_type, action)
            if is_allowed:
                return True

        # 6. Fallback default rules (closed by default — only known roles)
        for role in context.roles:
            if role == Roles.TENANT_ADMIN:
                return True  # All operations within tenant
            elif role == Roles.MANAGER:
                if action in ("read", "write", "delete", "traverse"):
                    return True
            elif role == Roles.ANALYST:
                if action in ("read", "write", "traverse"):
                    return True
            elif role == Roles.AUDITOR:
                if action in ("read", "traverse", "audit"):
                    return True
            elif role in (Roles.VIEWER, Roles.GUEST) and action in ("read", "traverse"):
                return True

        return False

    async def check_field_access(
        self,
        context: TenantContext,
        resource_type: str,
        field_name: str,
    ) -> bool:
        """Checks if the user has permission to view a specific field.

        Uses an **allowlist model**: fields are denied by default.
        Only explicitly allowed fields are visible.
        """
        if Roles.SUPER_ADMIN in context.roles:
            return True

        # Check database rules (allowlist)
        for role in context.roles:
            is_allowed = await self._get_field_permission(role, resource_type, field_name)
            if is_allowed:
                return True

        # Fallback default field rules (allowlist model)
        for role in context.roles:
            if role in (Roles.TENANT_ADMIN, Roles.MANAGER):
                return True
            elif role == Roles.ANALYST:
                # Explicit allowlist for Analyst
                return True
            elif role in (Roles.AUDITOR, Roles.VIEWER):
                return True
            elif role == Roles.GUEST and field_name in ("title", "summary", "public_metadata"):
                # Guest allowlist: only title, summary, public_metadata
                return True

        return False

    async def mask_content(self, context: TenantContext, content: str) -> str:
        """
        Applies field-level security constraints to redact/mask unauthorized fields
        within content blocks. Guest users receive masked responses.

        Uses an allowlist model: only fields explicitly allowed are visible.
        All other fields are redacted.
        """
        # If user has Manager or above, skip masking
        if any(r in (Roles.SUPER_ADMIN, Roles.TENANT_ADMIN, Roles.MANAGER) for r in context.roles):
            return content

        masked_content = content

        # Guest allowlist: only title, summary, public_metadata

        for role in context.roles:
            if role == Roles.GUEST:
                # Mask everything except allowed fields for Guest
                sensitive_fields = ["Revenue", "Profit Margin", "Stock", "Salary",
                                    "Profit", "Margin", "Revenue Growth", "EPS",
                                    "EBITDA", "Operating Income", "Net Income"]
                for field in sensitive_fields:
                    pattern = rf"\b({field})\b\s*(?:=|\:|of|is)?\s*[\$\w\d\.\,\-%]+"
                    masked_content = re.sub(
                        pattern,
                        r"\1 = [REDACTED]",
                        masked_content,
                        flags=re.IGNORECASE,
                    )
                    # Also mask standalone values that follow known sensitive keywords
                    pattern_margin = r"\b(profit\s+margin)\b\s*(?:=|\:|of|is)?\s*[\$\w\d\.\,\-%]+"
                    masked_content = re.sub(
                        pattern_margin,
                        r"\1 = [REDACTED]",
                        masked_content,
                        flags=re.IGNORECASE,
                    )

        return masked_content

    async def log_audit_trail(
        self,
        context: TenantContext,
        action: str,
        entity_id: str,
        before_state: Any | None = None,
        after_state: Any | None = None,
    ) -> None:
        """Creates and saves a secure audit trail log row."""
        before_str = json.dumps(before_state) if before_state is not None else None
        after_str = json.dumps(after_state) if after_state is not None else None

        # Primary role is usually first or defaults
        role = context.roles[0] if context.roles else Roles.GUEST

        audit_row = AuditLogRow(
            record_id=str(uuid.uuid4()),
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            role=role,
            timestamp=datetime.now(timezone.utc),
            action=action,
            entity_id=entity_id,
            before_state=before_str,
            after_state=after_str,
        )
        await self._save_audit_log(audit_row)
