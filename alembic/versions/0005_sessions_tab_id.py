"""
Sessions gain tab_id, unique per user.

Revision ID: 0005_sessions_tab_id
Revises: 0004_rules_is_placeholder
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_sessions_tab_id"
down_revision: str | None = "0004_rules_is_placeholder"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TAB_INDEX = "uq_sessions_user_tab"


def upgrade() -> None:
    # Nullable: a session does not have to belong to a tab, and the rows that exist when
    # this runs do not.
    op.add_column("sessions", sa.Column("tab_id", sa.Integer(), nullable=True))

    # One session per user per tab. `/context` fires on every DOM mutation a portal makes,
    # so without this two racing calls insert two rows and the 60-second cache reads
    # whichever it happens to find. Postgres treats NULLs as distinct by default, so rows
    # with no tab_id are unaffected.
    op.create_index(TAB_INDEX, "sessions", ["user_id", "tab_id"], unique=True)


def downgrade() -> None:
    op.drop_index(TAB_INDEX, table_name="sessions")
    op.drop_column("sessions", "tab_id")
