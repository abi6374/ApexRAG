"""
Add projections_json and metadata_json columns to apex_causal_edges
for unified Knowledge DAG edge storage.

Revision ID: 004
Revises: 003
Create Date: 2026-07-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add projections_json column — stores the DAG projection tags
    # (e.g. '["document"]', '["entity", "citation"]')
    op.add_column(
        "apex_causal_edges",
        sa.Column(
            "projections_json",
            sa.Text,
            nullable=False,
            server_default='["document"]',
        ),
    )

    # Add metadata_json column — stores arbitrary edge metadata
    op.add_column(
        "apex_causal_edges",
        sa.Column(
            "metadata_json",
            sa.Text,
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("apex_causal_edges", "projections_json")
    op.drop_column("apex_causal_edges", "metadata_json")
