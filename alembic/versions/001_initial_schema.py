"""
Initial ApexRAG schema — creates all core tables.

Revision ID: 001
Revises: None
Create Date: 2025-06-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── AST Nodes ───────────────────────────────────────────────────────
    op.create_table(
        "apex_ast_nodes",
        sa.Column("node_id", sa.String(36), primary_key=True),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("node_type", sa.String(20), nullable=False),
        sa.Column("depth", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "parent_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("children_json", sa.Text, nullable=False, server_default="[]"),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("source_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingestion_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("embedding_json", sa.Text, nullable=False, server_default="[]"),
    )
    op.create_index("ix_apex_nodes_doc", "apex_ast_nodes", ["doc_id"])
    op.create_index("ix_apex_nodes_tenant", "apex_ast_nodes", ["tenant_id"])
    op.create_index("ix_apex_nodes_parent", "apex_ast_nodes", ["parent_id"])
    op.create_index("ix_apex_nodes_type", "apex_ast_nodes", ["node_type"])

    # ── Temporal Metadata ───────────────────────────────────────────────
    op.create_table(
        "apex_temporal_metadata",
        sa.Column(
            "node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("source_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingestion_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("freshness_score", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("decay_rate", sa.Float, nullable=False, server_default="0.001"),
        sa.Column("superseded_by", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("revision_number", sa.Integer, nullable=False, server_default="0"),
        sa.Column("source_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="1"),
        sa.Column("previous_version", sa.String(36), nullable=True),
        sa.Column("validity_status", sa.String(20), nullable=False, server_default="ACTIVE"),
    )

    # ── Causal Edges ────────────────────────────────────────────────────
    op.create_table(
        "apex_causal_edges",
        sa.Column("edge_id", sa.String(36), primary_key=True),
        sa.Column(
            "source_node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("edge_type", sa.String(20), nullable=False),
        sa.Column("strength", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("evidence", sa.Text, nullable=False, server_default=""),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_apex_edges_source", "apex_causal_edges", ["source_node_id"])
    op.create_index("ix_apex_edges_target", "apex_causal_edges", ["target_node_id"])
    op.create_index("ix_apex_edges_type", "apex_causal_edges", ["edge_type"])

    # ── Page Index ──────────────────────────────────────────────────────
    op.create_table(
        "apex_page_index",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "node_id",
            sa.String(36),
            sa.ForeignKey("apex_ast_nodes.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("term", sa.String(512), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
    )
    op.create_index("ix_apex_pie_doc_term", "apex_page_index", ["doc_id", "term"])

    # ── Query Cache ─────────────────────────────────────────────────────
    op.create_table(
        "apex_query_cache",
        sa.Column("query_hash", sa.String(64), primary_key=True),
        sa.Column("doc_id", sa.String(255), primary_key=True),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("node_ids", sa.Text, nullable=False, server_default="[]"),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("apex_query_cache")
    op.drop_table("apex_page_index")
    op.drop_table("apex_causal_edges")
    op.drop_table("apex_temporal_metadata")
    op.drop_table("apex_ast_nodes")
