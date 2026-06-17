import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from apex_rag.ingestion.apex_storage import ApexStorage, AuditLogRow
from apex_rag.enterprise.auth.models import TenantContext
from apex_rag.enterprise.auth.access_control import AccessControlAgent, Roles
from apex_rag.agents.orchestrator import Orchestrator

@pytest.mark.asyncio
class TestRoleBasedAccessControl:

    @pytest.fixture
    async def storage(self) -> ApexStorage:
        return await ApexStorage.create("sqlite+aiosqlite:///:memory:")

    async def test_tenant_isolation(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        ctx_a = TenantContext(tenant_id="tenant_a", user_id="user-1", roles=[Roles.VIEWER])
        
        # Access document belonging to same tenant
        assert await agent.verify_tenant_access(ctx_a, "tenant_a") is True
        
        # Access document belonging to different tenant
        assert await agent.verify_tenant_access(ctx_a, "tenant_b") is False

        # SuperAdmin can cross tenant boundaries
        ctx_sa = TenantContext(tenant_id="tenant_a", user_id="user-sa", roles=[Roles.SUPER_ADMIN])
        assert await agent.verify_tenant_access(ctx_sa, "tenant_b") is True

    async def test_rbac_check_permissions(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        
        ctx_admin = TenantContext(tenant_id="tenant_a", user_id="admin-1", roles=[Roles.TENANT_ADMIN])
        ctx_viewer = TenantContext(tenant_id="tenant_a", user_id="viewer-1", roles=[Roles.VIEWER])
        ctx_guest = TenantContext(tenant_id="tenant_a", user_id="guest-1", roles=[Roles.GUEST])

        # TenantAdmin has all actions within tenant
        assert await agent.check_access(ctx_admin, "write", "document", "tenant_a") is True
        assert await agent.check_access(ctx_admin, "delete", "document", "tenant_a") is True

        # Viewer can read but not write/delete
        assert await agent.check_access(ctx_viewer, "read", "document", "tenant_a") is True
        assert await agent.check_access(ctx_viewer, "write", "document", "tenant_a") is False

        # Guest can read but not write/delete
        assert await agent.check_access(ctx_guest, "read", "document", "tenant_a") is True
        assert await agent.check_access(ctx_guest, "delete", "document", "tenant_a") is False

    async def test_field_level_security_masking(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        ctx_guest = TenantContext(tenant_id="tenant_a", user_id="guest-1", roles=[Roles.GUEST])
        ctx_manager = TenantContext(tenant_id="tenant_a", user_id="mgr-1", roles=[Roles.MANAGER])

        # Sensitive content
        content = "The company revenue is 120000 dollars and profit margin is 20% this quarter."

        # Guest user masking
        masked = await agent.mask_content(ctx_guest, content)
        assert "revenue = [REDACTED]" in masked.lower() or "revenue = [redacted]" in masked.lower() or "[redacted]" in masked.lower()
        assert "profit margin = [REDACTED]" in masked.lower() or "profit margin = [redacted]" in masked.lower() or "[redacted]" in masked.lower()
        assert "120000" not in masked

        # Manager user should not be masked
        unmasked = await agent.mask_content(ctx_manager, content)
        assert unmasked == content

    async def test_audit_trail_logging(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        ctx = TenantContext(tenant_id="tenant_a", user_id="user-1", roles=[Roles.ANALYST])

        # Log audit entry
        await agent.log_audit_trail(ctx, "READ_DOCUMENT", "doc-123", before_state={"status": "old"}, after_state={"status": "new"})

        # Query audit logs
        logs = await storage.get_audit_logs("tenant_a")
        assert len(logs) == 1
        assert logs[0].action == "READ_DOCUMENT"
        assert logs[0].entity_id == "doc-123"
        assert logs[0].role == Roles.ANALYST
        assert logs[0].user_id == "user-1"
        assert "status" in logs[0].after_state

    async def test_orchestrator_pre_retrieval_rbac(self) -> None:
        # Test Orchestrator pre-retrieval check stops run on unauthorized tenant
        from apex_rag.retrieval.agentic.navigator import ASTNavigationAgent
        planner = MagicMock()
        navigator = MagicMock()
        navigator._storage = MagicMock() # Will be MagicMock
        critic = MagicMock()

        orchestrator = Orchestrator(
            planner=planner,
            navigator=navigator,
            critic=critic
        )

        ctx = TenantContext(tenant_id="tenant_x", user_id="user-1", roles=[Roles.VIEWER])
        
        # When querying doc_id (which is checked as doc_tenant_id) differing from tenant_id, access is denied
        res = await orchestrator.execute_query("What is revenue?", "tenant_y", tenant_context=ctx)
        assert res is None

    async def test_custom_rbac_rules_and_evaluations(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        ctx = TenantContext(tenant_id="tenant_a", user_id="custom-1", roles=["CustomRole"])

        # Test custom action evaluator callback
        def custom_action_evaluator(context: TenantContext, resource_type: str) -> bool:
            return resource_type == "SpecialSecret" and "CustomRole" in context.roles

        agent.register_custom_execution("decrypt", custom_action_evaluator)
        assert await agent.check_access(ctx, "decrypt", "SpecialSecret") is True
        assert await agent.check_access(ctx, "decrypt", "GeneralFile") is False

        # Test dynamic database assignments
        await agent.assign_role_permission("CustomRole", "Config", "reboot", is_allowed=True)
        assert await agent.check_access(ctx, "reboot", "Config") is True

        await agent.assign_field_permission("CustomRole", "SecretNode", "classified_info", is_allowed=True)
        assert await agent.check_field_access(ctx, "SecretNode", "classified_info") is True

    async def test_dynamic_custom_rules_and_script_evaluators(self, storage: ApexStorage) -> None:
        agent = AccessControlAgent(storage)
        
        # 1. Test Custom Rule with expression: allow access if tenant is tenant_a
        await agent.define_custom_rule(
            name="TenantACheck",
            rule_type="expression",
            expression="context.tenant_id == 'tenant_a'",
            description="Allow only tenant_a"
        )
        
        # Assign rule to the Analyst role
        await agent.assign_custom_rule(rule_name="TenantACheck", role="Analyst", is_allowed=True)
        
        ctx_a = TenantContext(tenant_id="tenant_a", user_id="user-1", roles=["Analyst"])
        ctx_b = TenantContext(tenant_id="tenant_b", user_id="user-2", roles=["Analyst"])
        
        # Analyst on tenant_a is allowed, on tenant_b is not allowed
        assert await agent.check_access(ctx_a, "query", "ASTNode") is True
        assert await agent.check_access(ctx_b, "query", "ASTNode") is False
        
        # 2. Test Custom Rule with Python Script defining evaluate() function
        script_code = (
            "def evaluate(context, resource_type, action, env):\n"
            "    time_val = env.get('time')\n"
            "    if time_val:\n"
            "        return 9 <= time_val.hour < 17\n"
            "    return True\n"
        )
        await agent.define_custom_rule(
            name="BusinessHoursCheck",
            rule_type="script",
            expression=script_code,
            description="Allow only during business hours"
        )
        await agent.assign_custom_rule(rule_name="BusinessHoursCheck", role="Auditor", is_allowed=True)
        
        # Check during business hours (14:00) vs after hours (20:00)
        from datetime import datetime, timezone
        ctx_auditor = TenantContext(tenant_id="tenant_a", user_id="user-auditor", roles=["Auditor"])
        
        rule_def = await storage.get_custom_rule("BusinessHoursCheck")
        assert rule_def is not None
        
        # Business hours (14:00)
        env_day = {"time": datetime(2026, 6, 17, 14, 0, 0, tzinfo=timezone.utc)}
        assert agent.evaluate_custom_rule(rule_def, ctx_auditor, "ASTNode", "audit", env_day) is True
        
        # After hours (20:00)
        env_night = {"time": datetime(2026, 6, 17, 20, 0, 0, tzinfo=timezone.utc)}
        assert agent.evaluate_custom_rule(rule_def, ctx_auditor, "ASTNode", "audit", env_night) is False

        # 3. Test Custom Rule with Script setting 'result' variable
        script_result = "result = (context.user_id != 'user-1')"
        await agent.define_custom_rule(
            name="User1Only",
            rule_type="script",
            expression=script_result
        )
        await agent.assign_custom_rule(rule_name="User1Only", role="Guest", is_allowed=False)
        
        ctx_guest1 = TenantContext(tenant_id="tenant_a", user_id="user-1", roles=["Guest"])
        ctx_guest2 = TenantContext(tenant_id="tenant_a", user_id="user-2", roles=["Guest"])
        assert await agent.check_access(ctx_guest1, "read", "ASTNode") is True
        assert await agent.check_access(ctx_guest2, "read", "ASTNode") is False

        # 4. Test Precedence logic & Deny-Override
        # We have a rule that allows Analyst role (TenantACheck).
        # Now let's create a rule that denies specifically user-1.
        await agent.define_custom_rule(
            name="BlockUser1",
            rule_type="expression",
            expression="context.user_id == 'user-1'"
        )
        # Assign directly to user_id="user-1" with is_allowed=False
        await agent.assign_custom_rule(rule_name="BlockUser1", user_id="user-1", is_allowed=False)
        
        # user-1 has role Analyst. Even though TenantACheck allows Analyst, direct BlockUser1 is_allowed=False overrides!
        assert await agent.check_access(ctx_a, "query", "ASTNode") is False


