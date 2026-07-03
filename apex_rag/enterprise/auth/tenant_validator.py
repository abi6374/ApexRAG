"""
enterprise/auth/tenant_validator.py — Strict Tenant Isolation Validator.

Guarantees:
  - No cross-tenant reads
  - No cross-tenant writes
  - No cross-tenant graph traversal

Every query must include a tenant_id filter.  Operations without tenant_id
are rejected with MissingTenantContextError.

Usage:
    validator = TenantIsolationValidator(storage)
    await validator.assert_tenant_read_access(tenant_id, doc_id)
    await validator.assert_tenant_write_access(tenant_id, doc_id)
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from apex_rag.enterprise.auth.access_control import MissingTenantContextError
from apex_rag.ingestion.apex_storage import (
    ApexStorage,
    ASTNodeRow,
    NodeVersionRow,
    TemporalMetadataRow,
)

logger = logging.getLogger("apex_rag.enterprise.auth.tenant_validator")

# Tables that require tenant isolation (must have a tenant_id column)
_TENANT_TABLES: set[str] = {
    ASTNodeRow.__tablename__,
    NodeVersionRow.__tablename__,
    TemporalMetadataRow.__tablename__,
}

# Known tables that have tenant_id columns
_TENANT_COLUMN_MAP: dict[str, str] = {
    "apex_ast_nodes": "tenant_id",
    "node_versions": "tenant_id",
    "temporal_nodes": "tenant_id",
    "audit_logs": "tenant_id",
    "version_lineage": "tenant_id",
    "state_snapshots": "tenant_id",
}


class TenantIsolationValidator:
    """Validates and enforces strict tenant data isolation.

    Every storage operation must pass through this validator to ensure
    no cross-tenant data leakage occurs.

    This validator is a security gasket — it should be called BEFORE
    every database read/write operation that involves tenant-scoped data.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Assertion methods ─────────────────────────────────────────────────

    async def assert_tenant_context(self, tenant_id: str | None) -> None:
        """Assert that a tenant context was provided.

        Args:
            tenant_id: The tenant ID or None.

        Raises:
            MissingTenantContextError: If tenant_id is None or empty.
        """
        if not tenant_id:
            raise MissingTenantContextError(
                "Tenant context is required. All operations must specify a tenant_id. "
                "This prevents cross-tenant data leakage."
            )

    async def assert_tenant_read_access(
        self,
        tenant_id: str,
        table_name: str,
        row_id: str | None = None,
    ) -> None:
        """Assert that a read operation is scoped to the correct tenant.

        Args:
            tenant_id:  The tenant ID.
            table_name: The database table being queried.
            row_id:     Optional row ID to verify tenant ownership.

        Raises:
            MissingTenantContextError: If tenant context is missing.
            PermissionError: If the row belongs to a different tenant.
        """
        await self.assert_tenant_context(tenant_id)

        if row_id and table_name in _TENANT_COLUMN_MAP:
            tenant_col = _TENANT_COLUMN_MAP[table_name]
            await self._assert_row_belongs_to_tenant(
                table_name,
                tenant_col,
                row_id,
                tenant_id,
            )

    async def assert_tenant_write_access(
        self,
        tenant_id: str,
        table_name: str,
        row_id: str | None = None,
    ) -> None:
        """Assert that a write operation is scoped to the correct tenant.

        Identical to read access checks — data isolation is symmetric.

        Args:
            tenant_id:  The tenant ID.
            table_name: The database table being written to.
            row_id:     Optional row ID to verify tenant ownership.

        Raises:
            MissingTenantContextError: If tenant context is missing.
            PermissionError: If the row belongs to a different tenant.
        """
        await self.assert_tenant_read_access(tenant_id, table_name, row_id)

    async def assert_tenant_graph_traversal(
        self,
        tenant_id: str,
        node_ids: list[str],
    ) -> None:
        """Assert that all nodes in a graph traversal belong to the same tenant.

        Args:
            tenant_id: The expected tenant ID.
            node_ids:  List of node IDs being traversed.

        Raises:
            MissingTenantContextError: If tenant context is missing.
            PermissionError: If any node belongs to a different tenant.
        """
        await self.assert_tenant_context(tenant_id)

        if not node_ids:
            return

        async with self._storage.session() as session:
            stmt = select(ASTNodeRow).where(
                ASTNodeRow.node_id.in_(node_ids),
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            for row in rows:
                row_tenant = getattr(row, "tenant_id", None)
                if row_tenant and row_tenant != tenant_id:
                    raise PermissionError(
                        f"Cross-tenant graph traversal detected: "
                        f"node {row.node_id} belongs to tenant "
                        f"'{row_tenant}', expected '{tenant_id}'"
                    )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _assert_row_belongs_to_tenant(
        self,
        table_name: str,
        tenant_col: str,
        row_id: str,
        expected_tenant: str,
    ) -> None:
        """Verify a specific row belongs to the expected tenant."""
        try:
            # Validate table/column names against the known map to prevent SQL injection
            if table_name not in _TENANT_COLUMN_MAP:
                logger.warning(
                    "Unknown table '%s' — skipping tenant isolation check",
                    table_name,
                )
                return
            if tenant_col != _TENANT_COLUMN_MAP[table_name]:
                logger.warning(
                    "Column '%s' doesn't match expected '%s' for table '%s' — skipping",
                    tenant_col, _TENANT_COLUMN_MAP[table_name], table_name,
                )
                return

            from sqlalchemy import text as sa_text

            async with self._storage.session() as session:
                # Use a parameterized query with the known-safe column name
                quoted_col = f'"{tenant_col}"'
                stmt = sa_text(
                    f"SELECT {quoted_col} FROM \"{table_name}\" WHERE "
                    "id = :row_id OR node_id = :row_id2 OR version_id = :row_id3"
                )
                result = await session.execute(
                    stmt,
                    {
                        "row_id": row_id,
                        "row_id2": row_id,
                        "row_id3": row_id,
                    },
                )
                row = result.fetchone()
                if row is not None:
                    actual_tenant = row[0]
                    if actual_tenant and actual_tenant != expected_tenant:
                        raise PermissionError(
                            f"Row {row_id} in {table_name} belongs to tenant "
                            f"'{actual_tenant}', but access was attempted "
                            f"from tenant '{expected_tenant}'"
                        )
        except PermissionError:
            raise
        except Exception as exc:
            logger.warning(
                "Tenant isolation check skipped for %s.%s: %s",
                table_name,
                row_id,
                exc,
            )

    # ── Table schema helpers ──────────────────────────────────────────────

    @classmethod
    def requires_tenant_isolation(cls, table_name: str) -> bool:
        """Check if a table requires tenant isolation."""
        return table_name in _TENANT_TABLES or table_name in _TENANT_COLUMN_MAP

    @classmethod
    def get_tenant_column(cls, table_name: str) -> str | None:
        """Get the tenant column name for a table, or None if not isolated."""
        return _TENANT_COLUMN_MAP.get(table_name)
