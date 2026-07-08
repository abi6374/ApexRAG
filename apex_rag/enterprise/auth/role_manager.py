"""
enterprise/auth/role_manager.py — Role Manager for Custom Roles as Database Objects.

Provides CRUD operations for :class:`RoleProfile` objects stored in the database.
This is the authoritative service for managing custom roles — it replaces the
hardcoded ``ROLE_CONFIGS`` in ``RolePlannerAgent`` with database-backed profiles.

Usage:
    manager = RoleManager(storage=apex_storage)
    profile = await manager.create_role("ComplianceOfficer", tenant_id="tenant-1", created_by="admin")
    profile = await manager.get_role("ComplianceOfficer", "tenant-1")
    profiles = await manager.list_roles("tenant-1")
    await manager.delete_role("ComplianceOfficer", "tenant-1")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from apex_rag.enterprise.auth.models import RoleProfile
from apex_rag.ingestion.apex_storage import ApexStorage, RoleProfileRow


class RoleManager:
    """Service for CRUD operations on RoleProfile objects.

    All operations persist to the ``role_profiles`` table via ApexStorage.
    """

    def __init__(self, storage: ApexStorage) -> None:
        self._storage = storage

    # ── Create ─────────────────────────────────────────────────────────

    async def create_role(
        self,
        name: str,
        *,
        tenant_id: str = "default",
        created_by: str = "system",
        description: str = "",
        ranking_weights: dict[str, float] | None = None,
        visible_node_types: list[str] | None = None,
        hidden_node_types: list[str] | None = None,
        temporal_policy: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        field_visibility: dict[str, bool] | None = None,
        retrieval_preferences: dict[str, Any] | None = None,
    ) -> RoleProfile:
        """Create a new RoleProfile and persist it to the database.

        Args:
            name:                Role name (must be unique per tenant).
            tenant_id:           Tenant isolation boundary.
            created_by:          User ID or "system".
            description:         Optional description.
            ranking_weights:     ``{vector, keyword, structural}`` weights.
            visible_node_types:  If set, only these node types are visible.
            hidden_node_types:   Node types always hidden.
            temporal_policy:     Temporal constraint settings.
            allowed_tools:       List of tool/action names.
            field_visibility:    Dict of field_name → bool.
            retrieval_preferences: Retrieval mode preferences.

        Returns:
            The newly created :class:`RoleProfile`.

        Raises:
            ValueError: If a profile with the same name + tenant already exists.
        """
        existing = await self._storage.get_role_profile(name, tenant_id)
        if existing is not None:
            raise ValueError(
                f"Role profile '{name}' already exists for tenant '{tenant_id}'. "
                f"Use update_role() to modify it."
            )

        profile = RoleProfile(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            ranking_weights=ranking_weights or {
                "vector": 0.2, "keyword": 0.4, "structural": 0.4
            },
            visible_node_types=visible_node_types,
            hidden_node_types=hidden_node_types or [],
            temporal_policy=temporal_policy or {},
            allowed_tools=allowed_tools or ["read", "traverse", "search"],
            field_visibility=field_visibility or {},
            retrieval_preferences=retrieval_preferences or {},
            created_by=created_by,
            tenant_id=tenant_id,
            version=1,
            created_at=datetime.now(timezone.utc),
        )

        row = self._profile_to_row(profile)
        await self._storage.save_role_profile(row)
        return profile

    # ── Read ───────────────────────────────────────────────────────────

    async def get_role(
        self,
        name: str,
        tenant_id: str = "default",
    ) -> RoleProfile | None:
        """Fetch a RoleProfile by name + tenant_id.

        Returns:
            The :class:`RoleProfile` if found, or ``None``.
        """
        row = await self._storage.get_role_profile(name, tenant_id)
        if row is None:
            return None
        return self._row_to_profile(row)

    async def list_roles(self, tenant_id: str = "default") -> list[RoleProfile]:
        """List all RoleProfiles for a tenant.

        Returns:
            A list of :class:`RoleProfile` objects.
        """
        rows = await self._storage.list_role_profiles(tenant_id)
        return [self._row_to_profile(row) for row in rows]

    # ── Update ─────────────────────────────────────────────────────────

    async def update_role(
        self,
        name: str,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> RoleProfile:
        """Update an existing RoleProfile.

        Args:
            name:      The role name to update.
            tenant_id: Tenant isolation boundary.
            **kwargs:  Any fields from :class:`RoleProfile` to update.

        Returns:
            The updated :class:`RoleProfile`.

        Raises:
            ValueError: If the role profile doesn't exist.
        """
        row = await self._storage.get_role_profile(name, tenant_id)
        if row is None:
            raise ValueError(
                f"Role profile '{name}' not found for tenant '{tenant_id}'. "
                f"Use create_role() to create it first."
            )

        current = self._row_to_profile(row)
        updated = current.model_copy(update=kwargs)
        updated.version = current.version + 1

        updated_row = self._profile_to_row(updated)
        await self._storage.save_role_profile(updated_row)
        return updated

    # ── Delete ─────────────────────────────────────────────────────────

    async def delete_role(
        self,
        name: str,
        tenant_id: str = "default",
    ) -> bool:
        """Delete a RoleProfile by name + tenant_id.

        Returns:
            ``True`` if the profile was deleted.
        """
        return await self._storage.delete_role_profile(name, tenant_id)

    # ── Seed ───────────────────────────────────────────────────────────

    async def seed_default_roles(
        self,
        tenant_id: str = "default",
    ) -> list[RoleProfile]:
        """Seed the 7 default role profiles for a tenant.

        Called automatically when a new tenant is provisioned.
        Idempotent — existing profiles are not overwritten.

        Returns:
            The list of :class:`RoleProfile` objects for this tenant.
        """
        rows = await self._storage.seed_default_role_profiles(tenant_id)
        return [self._row_to_profile(row) for row in rows]

    # ── Converters ─────────────────────────────────────────────────────

    @staticmethod
    def _profile_to_row(profile: RoleProfile) -> RoleProfileRow:
        """Convert a Pydantic RoleProfile to an ORM RoleProfileRow."""
        return RoleProfileRow(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            ranking_weights=json.dumps(profile.ranking_weights),
            visible_node_types=json.dumps(profile.visible_node_types)
            if profile.visible_node_types is not None
            else None,
            hidden_node_types=json.dumps(profile.hidden_node_types),
            temporal_policy=json.dumps(profile.temporal_policy),
            allowed_tools=json.dumps(profile.allowed_tools),
            field_visibility=json.dumps(profile.field_visibility),
            retrieval_preferences=json.dumps(profile.retrieval_preferences),
            created_by=profile.created_by,
            tenant_id=profile.tenant_id,
            version=profile.version,
            created_at=profile.created_at,
        )

    @staticmethod
    def _row_to_profile(row: RoleProfileRow) -> RoleProfile:
        """Convert an ORM RoleProfileRow to a Pydantic RoleProfile."""
        return RoleProfile(
            id=row.id,
            name=row.name,
            description=row.description,
            ranking_weights=json.loads(row.ranking_weights) if row.ranking_weights else {},
            visible_node_types=json.loads(row.visible_node_types)
            if row.visible_node_types
            else None,
            hidden_node_types=json.loads(row.hidden_node_types) if row.hidden_node_types else [],
            temporal_policy=json.loads(row.temporal_policy) if row.temporal_policy else {},
            allowed_tools=json.loads(row.allowed_tools) if row.allowed_tools else [],
            field_visibility=json.loads(row.field_visibility) if row.field_visibility else {},
            retrieval_preferences=json.loads(row.retrieval_preferences)
            if row.retrieval_preferences
            else {},
            created_by=row.created_by,
            tenant_id=row.tenant_id,
            version=row.version,
            created_at=row.created_at,
        )
