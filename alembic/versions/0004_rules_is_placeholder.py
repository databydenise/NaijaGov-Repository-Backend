"""
Rules gain is_placeholder.

Revision ID: 0004_rules_is_placeholder
Revises: 0003_token_last4
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_rules_is_placeholder"
down_revision: str | None = "0003_token_last4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Flags invented demo content so it can never be presented as official guidance.
    # The default is false, not true: an existing row predates the placeholder set, and a
    # new one has to opt in to being demo content rather than opt out of it.
    op.add_column(
        "rules",
        sa.Column(
            "is_placeholder",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("rules", "is_placeholder")
