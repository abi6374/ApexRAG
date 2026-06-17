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

