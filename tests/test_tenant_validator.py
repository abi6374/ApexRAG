"""
tests/test_tenant_validator.py — Tests for the TenantIsolationValidator (Phase 5).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from apex_rag.enterprise.auth.access_control import MissingTenantContextError
from apex_rag.enterprise.auth.tenant_validator import TenantIsolationValidator


@pytest.fixture
def mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.is_sqlite = True

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    storage.session.return_value = mock_session

    return storage


@pytest.fixture
def validator(mock_storage: MagicMock) -> TenantIsolationValidator:
    return TenantIsolationValidator(mock_storage)


class TestTenantIsolationValidator:
    """Tests for the strict tenant isolation validator."""

    @pytest.mark.asyncio
    async def test_assert_tenant_context_missing(self, validator: TenantIsolationValidator) -> None:
        """Should raise MissingTenantContextError when tenant_id is None."""
        with pytest.raises(MissingTenantContextError):
            await validator.assert_tenant_context(None)

    @pytest.mark.asyncio
    async def test_assert_tenant_context_empty(self, validator: TenantIsolationValidator) -> None:
        """Should raise MissingTenantContextError when tenant_id is empty."""
        with pytest.raises(MissingTenantContextError):
            await validator.assert_tenant_context("")

    @pytest.mark.asyncio
    async def test_assert_tenant_context_valid(self, validator: TenantIsolationValidator) -> None:
        """Should not raise when tenant_id is provided."""
        await validator.assert_tenant_context("tenant-a")  # Should not raise

    @pytest.mark.asyncio
    async def test_assert_tenant_read_access_with_context(
        self, validator: TenantIsolationValidator
    ) -> None:
        """Should pass with valid tenant context (no row_id)."""
        await validator.assert_tenant_read_access("tenant-a", "apex_ast_nodes")

    @pytest.mark.asyncio
    async def test_assert_tenant_read_access_missing_context(
        self, validator: TenantIsolationValidator
    ) -> None:
        """Should raise when tenant context is missing."""
        with pytest.raises(MissingTenantContextError):
            await validator.assert_tenant_read_access("", "apex_ast_nodes")

    @pytest.mark.asyncio
    async def test_assert_write_access(self, validator: TenantIsolationValidator) -> None:
        """Should handle write access checks (same as read)."""
        await validator.assert_tenant_write_access("tenant-a", "apex_ast_nodes")

    @pytest.mark.asyncio
    async def test_requires_tenant_isolation(self) -> None:
        """Should correctly identify tables requiring isolation."""
        assert TenantIsolationValidator.requires_tenant_isolation("apex_ast_nodes") is True
        assert TenantIsolationValidator.requires_tenant_isolation("node_versions") is True
        assert TenantIsolationValidator.requires_tenant_isolation("non_existent_table") is False

    def test_get_tenant_column(self) -> None:
        """Should return correct tenant column names."""
        assert TenantIsolationValidator.get_tenant_column("apex_ast_nodes") == "tenant_id"
        assert TenantIsolationValidator.get_tenant_column("non_existent") is None

    @pytest.mark.asyncio
    async def test_assert_graph_traversal_empty(self, validator: TenantIsolationValidator) -> None:
        """Should pass with empty node_ids list."""
        await validator.assert_tenant_graph_traversal("tenant-a", [])
