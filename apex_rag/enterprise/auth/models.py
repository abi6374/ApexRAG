from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    """
    Enterprise Multi-Tenant Context.
    Passed through the entire application stack to guarantee data isolation.
    """

    tenant_id: str = Field(..., description="Unique identifier for the workspace or tenant")
    user_id: str = Field(..., description="ID of the user making the request")
    roles: list[str] = Field(
        default_factory=list, description="RBAC roles (e.g. 'admin', 'reader')"
    )


class APIKey(BaseModel):
    """
    Representation of an authenticated API Key.
    """

    key_hash: str
    tenant_id: str
    is_active: bool = True


class RoleProfile(BaseModel):
    """
    A custom role stored as a database object, not a Python class.

    Every tenant can define unlimited custom roles with fine-grained
    control over retrieval, visibility, and reasoning behaviour.

    Fields mirror the vision document specification:
        - ``id``: Auto-generated UUID4
        - ``name``: Human-readable role name (e.g. "ComplianceOfficer")
        - ``description``: Optional description of the role's purpose
        - ``ranking_weights``: ``{vector, keyword, structural}`` weights in [0,1]
        - ``visible_node_types``: If set, only these node types are visible
        - ``hidden_node_types``: Node types always hidden from this role
        - ``temporal_policy``: Dict of temporal constraints
        - ``allowed_tools``: List of tool/action names this role can use
        - ``field_visibility``: Dict of field_name → bool (field-level masking)
        - ``retrieval_preferences``: Dict of retrieval mode preferences
        - ``created_by``: User ID that created this role
        - ``tenant_id``: Tenant isolation boundary
        - ``version``: Monotonically increasing version counter
        - ``created_at``: UTC timestamp
    """

    # ── Identity ─────────────────────────────────────────────────────
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""

    # ── RetrieWeights (applied by RolePlannerAgent) ──────────────────
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {"vector": 0.2, "keyword": 0.4, "structural": 0.4}
    )

    # ── Visibility ───────────────────────────────────────────────────
    visible_node_types: list[str] | None = Field(
        default=None, description="None = all node types visible"
    )
    hidden_node_types: list[str] = Field(default_factory=list)

    # ── Temporal ─────────────────────────────────────────────────────
    temporal_policy: dict[str, Any] = Field(default_factory=dict)

    # ── Security ─────────────────────────────────────────────────────
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["read", "traverse", "search"]
    )
    field_visibility: dict[str, bool] = Field(
        default_factory=dict,
        description="Field-level visibility overrides. Key = field name, value = visible",
    )

    # ── Retrieval Preferences ────────────────────────────────────────
    retrieval_preferences: dict[str, Any] = Field(default_factory=dict)

    # ── Audit & Versioning ───────────────────────────────────────────
    created_by: str = "system"
    tenant_id: str = "default"
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Property helpers ─────────────────────────────────────────────

    @property
    def effective_visible_types(self) -> list[str] | None:
        """Return visible_node_types, or generate from hidden types if visible not set."""
        if self.visible_node_types is not None:
            return self.visible_node_types
        return None

    def to_config_dict(self) -> dict[str, Any]:
        """Convert to the config dict format expected by RolePlannerAgent."""
        return {
            "retrieval_preferences": self.retrieval_preferences,
            "ranking_weights": self.ranking_weights,
            "hidden_node_types": self.hidden_node_types,
            "visible_node_types": self.visible_node_types,
            "applied_policies": [f"role_profile:{self.name}"],
        }

    model_config = {"use_enum_values": True}
