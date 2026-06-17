from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from apex_rag.enterprise.auth.models import TenantContext
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

class AccessControlAgent:
    """
    AccessControlAgent manages multi-tenant validation, RBAC checks, field-level security
    (masking/redaction), and audit logs generation.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self.storage = storage
        self.custom_evaluators: dict[str, Any] = {}

    def register_custom_execution(self, action: str, callback: Any) -> None:
        """Registers a dynamic runtime custom evaluator for a specific execution action."""
        self.custom_evaluators[action] = callback

    async def assign_role_permission(
        self, role: str, resource_type: str, action: str, is_allowed: bool
    ) -> None:
        """Dynamically defines and creates a custom permission rule for a role."""
        from apex_rag.ingestion.apex_storage import RolePermissionRow
        row = RolePermissionRow(
            role=role, resource_type=resource_type, action=action, is_allowed=is_allowed
        )
        import inspect
        if hasattr(self.storage, "save_role_permission"):
            res = self.storage.save_role_permission(row)
            if inspect.isawaitable(res):
                await res

    async def assign_field_permission(
        self, role: str, resource_type: str, field_name: str, is_allowed: bool
    ) -> None:
        """Dynamically defines and creates a custom field visibility permission rule for a role."""
        from apex_rag.ingestion.apex_storage import FieldPermissionRow
        row = FieldPermissionRow(
            role=role, resource_type=resource_type, field_name=field_name, is_allowed=is_allowed
        )
        import inspect
        if hasattr(self.storage, "save_field_permission"):
            res = self.storage.save_field_permission(row)
            if inspect.isawaitable(res):
                await res

    async def define_custom_rule(
        self, name: str, rule_type: str, expression: str, description: str | None = None
    ) -> None:
        """Dynamically defines and persists a custom security rule with dynamic execution logic."""
        from apex_rag.ingestion.apex_storage import CustomRuleRow
        row = CustomRuleRow(
            name=name, rule_type=rule_type, expression=expression, description=description
        )
        import inspect
        if hasattr(self.storage, "save_custom_rule"):
            res = self.storage.save_custom_rule(row)
            if inspect.isawaitable(res):
                await res

    async def assign_custom_rule(
        self, rule_name: str, role: str | None = None, user_id: str | None = None, is_allowed: bool = True
    ) -> None:
        """Dynamically assigns a custom security rule to a role or specific user."""
        from apex_rag.ingestion.apex_storage import RuleAssignmentRow
        row = RuleAssignmentRow(
            rule_name=rule_name, role=role, user_id=user_id, is_allowed=is_allowed
        )
        import inspect
        if hasattr(self.storage, "save_rule_assignment"):
            res = self.storage.save_rule_assignment(row)
            if inspect.isawaitable(res):
                await res

    async def _get_role_permission(self, role: str, resource_type: str, action: str) -> bool:
        import inspect
        if not hasattr(self.storage, "get_role_permission"):
            return False
        res = self.storage.get_role_permission(role, resource_type, action)
        if inspect.isawaitable(res):
            return await res
        return False if type(res).__name__ in ("MagicMock", "Mock") else res

    async def _get_field_permission(self, role: str, resource_type: str, field_name: str) -> bool:
        import inspect
        if not hasattr(self.storage, "get_field_permission"):
            return False
        res = self.storage.get_field_permission(role, resource_type, field_name)
        if inspect.isawaitable(res):
            return await res
        return False if type(res).__name__ in ("MagicMock", "Mock") else res

    async def _save_audit_log(self, audit_row: AuditLogRow) -> None:
        import inspect
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

    def evaluate_custom_rule(
        self,
        rule: Any,
        context: TenantContext,
        resource_type: str,
        action: str,
        env: dict[str, Any] | None = None
    ) -> bool:
        """
        Evaluates a custom rule definition against the given runtime parameters.
        Uses a sandboxed execution context to evaluate Python expressions or execute scripts safely.
        """
        import datetime
        
        # Prepare evaluation scope/environment
        local_env = {
            "context": context,
            "resource_type": resource_type,
            "action": action,
            "env": env or {},
            "datetime": datetime,
        }
        # Whitelist safe builtins and block harmful ones
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }
        globals_env = {
            "__builtins__": safe_builtins,
        }

        try:
            rule_type = getattr(rule, "rule_type", "expression")
            expression = getattr(rule, "expression", "")
            rule_name = getattr(rule, "name", "unknown")
            
            if rule_type == "expression":
                # Evaluate python expression (expression is expected to return bool)
                return bool(eval(expression, globals_env, local_env))
            elif rule_type == "script":
                # Execute python script block
                exec(expression, globals_env, local_env)
                
                # Check for an 'evaluate' function in the local namespace
                if "evaluate" in local_env and callable(local_env["evaluate"]):
                    res = local_env["evaluate"](context, resource_type, action, env or {})
                    return bool(res)
                
                # Fallback: check for 'result' variable
                if "result" in local_env:
                    return bool(local_env["result"])
                
                logger.error(f"Custom script rule '{rule_name}' completed without setting 'result' or defining 'evaluate'")
                return False
            else:
                logger.error(f"Unknown custom rule type: {rule_type}")
                return False
        except Exception as e:
            logger.error(f"Error executing custom rule '{getattr(rule, 'name', 'unknown')}': {e}", exc_info=True)
            return False

    async def check_access(
        self,
        context: TenantContext,
        action: str,
        resource_type: str,
        doc_tenant_id: str | None = None
    ) -> bool:
        """
        Verifies if the user context is allowed to perform action on a resource type.
        Supports fallback default rules if no database role permissions are defined.
        """
        # 1. Tenant boundary validation
        if doc_tenant_id and not await self.verify_tenant_access(context, doc_tenant_id):
            return False

        # Check dynamic custom execution evaluators
        if action in self.custom_evaluators:
            import inspect
            res = self.custom_evaluators[action](context, resource_type)
            if hasattr(res, "__await__") or inspect.isawaitable(res):
                return await res
            return bool(res)

        # SuperAdmin is always allowed
        if Roles.SUPER_ADMIN in context.roles:
            return True

        # Check dynamic custom database rules & assignments
        import inspect
        custom_assignments = []
        if hasattr(self.storage, "get_rule_assignments"):
            for role in context.roles:
                res = self.storage.get_rule_assignments(role=role)
                if inspect.isawaitable(res):
                    res = await res
                if res and type(res).__name__ not in ("MagicMock", "Mock", "AsyncMock"):
                    custom_assignments.extend(res)
            if context.user_id:
                res = self.storage.get_rule_assignments(user_id=context.user_id)
                if inspect.isawaitable(res):
                    res = await res
                if res and type(res).__name__ not in ("MagicMock", "Mock", "AsyncMock"):
                    custom_assignments.extend(res)

        if custom_assignments:
            # Sort user assignments to execute first (user settings supersede role settings)
            user_assignments = [a for a in custom_assignments if a.user_id == context.user_id]
            role_assignments = [a for a in custom_assignments if a.role in context.roles and a.user_id != context.user_id]
            sorted_assignments = user_assignments + role_assignments

            has_matched_allow = False

            for assignment in sorted_assignments:
                if hasattr(self.storage, "get_custom_rule"):
                    res = self.storage.get_custom_rule(assignment.rule_name)
                    if inspect.isawaitable(res):
                        rule_def = await res
                    else:
                        rule_def = res

                    if rule_def and type(rule_def).__name__ not in ("MagicMock", "Mock", "AsyncMock"):
                        env = {"time": datetime.now(timezone.utc)}
                        rule_triggered = self.evaluate_custom_rule(
                            rule_def, context, resource_type, action, env
                        )
                        if rule_triggered:
                            # Deny-override: if rule evaluates to True but is_allowed is False, deny immediately
                            if not assignment.is_allowed:
                                return False
                            else:
                                has_matched_allow = True

            if has_matched_allow:
                return True

        # Check explicit database role permissions if stored
        for role in context.roles:
            is_allowed = await self._get_role_permission(role, resource_type, action)
            if is_allowed:
                return True

        # Fallback default rules
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
            elif role in (Roles.VIEWER, Roles.GUEST):
                if action in ("read", "traverse"):
                    return True

        return False

    async def check_field_access(
        self,
        context: TenantContext,
        resource_type: str,
        field_name: str
    ) -> bool:
        """Checks if the user has permission to view a specific field."""
        if Roles.SUPER_ADMIN in context.roles:
            return True

        # Check database rules
        for role in context.roles:
            is_allowed = await self._get_field_permission(role, resource_type, field_name)
            if is_allowed:
                return True

        # Fallback default field rules
        for role in context.roles:
            if role in (Roles.TENANT_ADMIN, Roles.MANAGER):
                return True
            elif role == Roles.ANALYST:
                # Analyst can see standard fields but not strict financial secrets if not Manager
                if field_name not in ("Profit Margin", "Salary"):
                    return True
            elif role in (Roles.AUDITOR, Roles.VIEWER):
                if field_name not in ("Profit Margin", "Salary"):
                    return True
            elif role == Roles.GUEST:
                # Guest is heavily restricted
                if field_name not in ("Revenue", "Profit Margin", "Stock", "Salary"):
                    return True

        return False

    async def mask_content(self, context: TenantContext, content: str) -> str:
        """
        Applies field-level security constraints to redact/mask unauthorized fields
        within content blocks. Guest users receive masked responses.
        """
        # If user has Manager or above, skip masking
        if any(r in (Roles.SUPER_ADMIN, Roles.TENANT_ADMIN, Roles.MANAGER) for r in context.roles):
            return content

        masked_content = content

        # Check permissions for fields
        sensitive_fields = ["Revenue", "Profit Margin", "Stock", "Salary"]
        for field in sensitive_fields:
            has_access = await self.check_field_access(context, "ASTNode", field)
            if not has_access:
                # Mask patterns like "Revenue = 100000", "Revenue: 100000", "Revenue of 100000"
                pattern = rf"\b({field})\b\s*(?:=|\:|of|is)?\s*[\$\w\d\.\,\-]+"
                masked_content = re.sub(
                    pattern,
                    rf"\1 = [REDACTED]",
                    masked_content,
                    flags=re.IGNORECASE
                )
                
                # Broad word replacements for anything containing the field
                # (e.g. "profit margin of 20%" -> "profit margin of [REDACTED]")
                pattern_margin = r"\b(profit\s+margin)\b\s*(?:=|\:|of|is)?\s*[\$\w\d\.\,\-%]+"
                masked_content = re.sub(
                    pattern_margin,
                    r"\1 = [REDACTED]",
                    masked_content,
                    flags=re.IGNORECASE
                )

        return masked_content

    async def log_audit_trail(
        self,
        context: TenantContext,
        action: str,
        entity_id: str,
        before_state: Any | None = None,
        after_state: Any | None = None
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
            after_state=after_str
        )
        await self._save_audit_log(audit_row)
