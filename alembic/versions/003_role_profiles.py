"""
Add role_profiles table for Custom Roles as Database Objects.

Revision ID: 003
Revises: 002
Create Date: 2026-07-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "role_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "ranking_weights",
            sa.Text,
            nullable=False,
            server_default='{"vector": 0.2, "keyword": 0.4, "structural": 0.4}',
        ),
        sa.Column("visible_node_types", sa.Text, nullable=True),
        sa.Column("hidden_node_types", sa.Text, nullable=False, server_default="[]"),
        sa.Column("temporal_policy", sa.Text, nullable=False, server_default="{}"),
        sa.Column(
            "allowed_tools",
            sa.Text,
            nullable=False,
            server_default='["read", "traverse", "search"]',
        ),
        sa.Column("field_visibility", sa.Text, nullable=False, server_default="{}"),
        sa.Column("retrieval_preferences", sa.Text, nullable=False, server_default="{}"),
        sa.Column("created_by", sa.String(255), nullable=False, server_default="system"),
        sa.Column("tenant_id", sa.String(255), nullable=False, server_default="default", index=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_role_profiles_name", "role_profiles", ["name"])
    op.create_index("ix_role_profiles_tenant", "role_profiles", ["tenant_id"])
    op.create_index(
        "ix_role_profiles_name_tenant",
        "role_profiles",
        ["name", "tenant_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("role_profiles")
