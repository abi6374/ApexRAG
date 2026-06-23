"""
enterprise/auth/policy_engine.py — Deterministic policy engine.

Replaces the insecure eval()/exec() custom rule evaluator with a
type-safe, operator-based policy evaluation system.  Supports all
standard comparison and membership operators without dynamic code execution.

Usage:
    rule = PolicyRule(
        field="department",
        operator="EQ",
        value="Finance"
    )
    engine = PolicyEngine()
    result = engine.evaluate(rule, context={"department": "Finance"})  # True
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("apex_rag.enterprise.auth.policy_engine")

# ── Supported Operators ─────────────────────────────────────────────────────

SUPPORTED_OPERATORS: dict[str, Any] = {
    "EQ": operator.eq,
    "NE": operator.ne,
    "GT": operator.gt,
    "LT": operator.lt,
    "GTE": operator.ge,
    "LTE": operator.le,
    "IN": lambda a, b: a in b,
    "NOT_IN": lambda a, b: a not in b,
    "CONTAINS": lambda a, b: b in a if isinstance(a, (str, list, tuple, dict)) else False,
    "STARTS_WITH": lambda a, b: isinstance(a, str) and a.startswith(b),
    "ENDS_WITH": lambda a, b: isinstance(a, str) and a.endswith(b),
}

__all__ = [
    "PolicyRule",
    "PolicyCondition",
    "PolicyEvaluator",
    "PolicyEngine",
    "SUPPORTED_OPERATORS",
]


# ── Domain Models ───────────────────────────────────────────────────────────


@dataclass
class PolicyRule:
    """A single atomic policy rule — field + operator + expected value.

    Example:
        PolicyRule(field="role", operator="IN", value=["admin", "manager"])
    """

    field: str
    operator: str
    value: Any
    description: str = ""


@dataclass
class PolicyCondition:
    """A compound condition composed of multiple PolicyRules.

    ``match`` can be "ALL" (AND logic) or "ANY" (OR logic).

    Example:
        PolicyCondition(
            rules=[
                PolicyRule("role", "IN", ["admin", "manager"]),
                PolicyRule("department", "EQ", "Finance"),
            ],
            match="ALL",
        )
    """

    rules: list[PolicyRule] = field(default_factory=list)
    match: str = "ALL"  # "ALL" (AND) | "ANY" (OR)


@dataclass
class PolicyEvaluator:
    """Evaluates a single PolicyRule against a context dictionary.

    Thread-safe and deterministic — no dynamic code execution.
    """

    @classmethod
    def evaluate_rule(cls, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Evaluate a single rule against a context dictionary.

        Args:
            rule:    The PolicyRule to evaluate.
            context: Dictionary of field_name -> value.

        Returns:
            True if the rule matches, False otherwise.

        Raises:
            ValueError: If the operator is unknown.
        """
        op_func = SUPPORTED_OPERATORS.get(rule.operator.upper())
        if op_func is None:
            raise ValueError(
                f"Unknown operator '{rule.operator}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_OPERATORS))}"
            )

        actual_value = context.get(rule.field)
        try:
            return bool(op_func(actual_value, rule.value))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Policy evaluation failed for rule %s: %s",
                rule.field,
                exc,
            )
            return False

    @classmethod
    def evaluate_condition(
        cls, condition: PolicyCondition, context: dict[str, Any]
    ) -> bool:
        """Evaluate a compound condition (ALL = AND, ANY = OR).

        Args:
            condition: The PolicyCondition to evaluate.
            context:   Dictionary of field_name -> value.

        Returns:
            True if the condition matches.
        """
        if not condition.rules:
            return True

        results = [
            cls.evaluate_rule(rule, context) for rule in condition.rules
        ]

        if condition.match == "ALL":
            return all(results)
        elif condition.match == "ANY":
            return any(results)
        else:
            logger.error("Unknown match mode '%s', defaulting to ALL", condition.match)
            return all(results)


# ── Policy Engine ───────────────────────────────────────────────────────────


class PolicyEngine:
    """Central policy engine for role-based and user-based access control.

    Evaluates policies composed of PolicyRules/PolicyConditions against
    runtime context (tenant, roles, resource type, action, environment).

    No eval(), no exec(), no dynamic Python execution of any kind.
    """

    def __init__(self) -> None:
        self._rules: dict[str, PolicyCondition] = {}
        self._assignments: list[tuple[str, str, bool]] = []  # (rule_name, target, is_allowed)

    def add_policy(self, name: str, condition: PolicyCondition) -> None:
        """Register a named policy condition.

        Args:
            name:      Unique policy name.
            condition: The PolicyCondition to evaluate.
        """
        self._rules[name] = condition

    def remove_policy(self, name: str) -> None:
        """Remove a previously registered policy by name."""
        self._rules.pop(name, None)

    def get_policy(self, name: str) -> PolicyCondition | None:
        """Retrieve a policy condition by name."""
        return self._rules.get(name)

    def assign_policy_to_role(
        self, rule_name: str, role: str, is_allowed: bool = True
    ) -> None:
        """Assign a policy to a role.

        Args:
            rule_name:  Name of the registered policy.
            role:       Role name (e.g. "Manager", "Analyst").
            is_allowed: Whether this assignment grants or denies access.
        """
        self._assignments.append((rule_name, f"role:{role}", is_allowed))

    def assign_policy_to_user(
        self, rule_name: str, user_id: str, is_allowed: bool = True
    ) -> None:
        """Assign a policy directly to a user.

        User assignments take precedence over role assignments.

        Args:
            rule_name:  Name of the registered policy.
            user_id:    The user ID.
            is_allowed: Whether this assignment grants or denies access.
        """
        self._assignments.append((rule_name, f"user:{user_id}", is_allowed))

    def evaluate(
        self,
        context: dict[str, Any],
        *,
        roles: list[str] | None = None,
        user_id: str | None = None,
    ) -> tuple[bool, bool]:
        """Evaluate all applicable policies for a given context.

        User-level assignments are evaluated before role-level assignments
        (user policies take precedence over role policies).

        If a policy condition matches and the assignment is ``is_allowed=False``,
        access is immediately denied (deny-override).  If any matching policy
        grants access, ``True`` is returned.

        Args:
            context: Runtime context dictionary (tenant_id, resource_type,
                     action, env, etc.).
            roles:   Optional list of role names to filter assignments.
            user_id: Optional user ID to filter assignments (user policies
                     evaluated first, take precedence).

        Returns:
            A tuple of ``(result, applied)`` where:
            - ``result`` is ``True`` if access is granted, ``False`` if denied.
            - ``applied`` is ``True`` if any policies were actually evaluated
              (i.e. there were applicable assignments). When ``applied`` is
              ``False``, the caller should fall through to other authorization
              mechanisms.
        """
        # Collect applicable assignments
        applicable: list[tuple[str, bool, str]] = []  # (rule_name, is_allowed, target_type)

        for rule_name, target, is_allowed in self._assignments:
            target_type, target_value = target.split(":", 1)
            if target_type == "user" and user_id and target_value == user_id:
                applicable.append((rule_name, is_allowed, "user"))
            elif target_type == "role" and roles and target_value in roles:
                applicable.append((rule_name, is_allowed, "role"))

        if not applicable:
            return True, False  # No policies applied — caller should fall through

        # Sort: user assignments first (higher priority)
        applicable.sort(key=lambda a: (0 if a[2] == "user" else 1))

        has_allow = False
        for rule_name, is_allowed, _target_type in applicable:
            condition = self._rules.get(rule_name)
            if condition is None:
                logger.warning("Policy '%s' not found, skipping", rule_name)
                continue

            if PolicyEvaluator.evaluate_condition(condition, context):
                if not is_allowed:
                    return False, True  # Explicit deny takes precedence
                has_allow = True

        return has_allow, True  # Policies were applied

    def clear(self) -> None:
        """Remove all policies and assignments (for testing/reset)."""
        self._rules.clear()
        self._assignments.clear()


# ── Convenience factory ─────────────────────────────────────────────────────


def make_policy_rule(
    field: str,
    operator: str,
    value: Any,
    *,
    description: str = "",
) -> PolicyRule:
    """Create a single PolicyRule with validation.

    Args:
        field:       Context field name (e.g. "department", "role").
        operator:    One of the SUPPORTED_OPERATORS keys.
        value:       Expected value to compare against.
        description: Optional human-readable description.

    Returns:
        A PolicyRule instance.

    Raises:
        ValueError: If the operator is not supported.
    """
    op_upper = operator.upper()
    if op_upper not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported operator '{operator}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_OPERATORS))}"
        )
    return PolicyRule(field=field, operator=op_upper, value=value, description=description)
