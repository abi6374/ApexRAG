"""
Create node_versions table with FK constraints and content_hash.

Revision ID: 002
Revises: 001
Create Date: 2025-06-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "node_versions",
        sa.Column("version_id", sa.String(36), primary_key=True),
        sa.Column(
            "node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("content_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("revision_number", sa.Integer, nullable=False, server_default="0"),
        sa.Column("source_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="1"),
        sa.Column(
            "superseded_by",
            sa.String(36),
            sa.ForeignKey("node_versions.version_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "previous_version",
            sa.String(36),
            sa.ForeignKey("node_versions.version_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("validity_status", sa.String(20), nullable=False, server_default="ACTIVE"),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False, server_default="default"),
    )
    op.create_index("ix_node_versions_node_id", "node_versions", ["node_id"])
    op.create_index("ix_node_versions_doc", "node_versions", ["doc_id"])
    op.create_index("ix_node_versions_tenant", "node_versions", ["tenant_id"])
    op.create_index("ix_node_versions_current", "node_versions", ["node_id", "is_current"])
    op.create_index(
        "ix_node_versions_effective", "node_versions", ["node_id", "effective_from", "effective_to"]
    )

    # ── Version Lineage ─────────────────────────────────────────────────
    op.create_table(
        "version_lineage",
        sa.Column("lineage_id", sa.String(36), primary_key=True),
        sa.Column(
            "node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column(
            "source_version_id",
            sa.String(36),
            sa.ForeignKey("node_versions.version_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_version_id",
            sa.String(36),
            sa.ForeignKey("node_versions.version_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("lineage_type", sa.String(30), nullable=False, server_default="VERSION_OF"),
        sa.Column("strength", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("evidence", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_vl_node", "version_lineage", ["node_id"])
    op.create_index("ix_vl_source", "version_lineage", ["source_version_id"])
    op.create_index("ix_vl_target", "version_lineage", ["target_version_id"])
    op.create_index("ix_vl_type", "version_lineage", ["lineage_type"])


def downgrade() -> None:
    op.drop_table("version_lineage")
    op.drop_table("node_versions")
