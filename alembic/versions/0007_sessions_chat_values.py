"""
Sessions gain chat_values.

Revision ID: 0007_sessions_chat_values
Revises: 0006_documents
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_sessions_chat_values"
down_revision: str | None = "0006_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # What the user has told us in chat, as profile-shaped keys: `{"lga": "Ikeja"}`. Held
    # apart from `history` because the two are read differently — history is prose for the
    # prompt, this is the set of values a fill may resolve against — and because a value
    # given ten turns ago must outlive the turn that carried it.
    #
    # Never the profile: there is no consent screen behind a chat message, so a value given
    # here lives exactly as long as the session and dies with it.
    op.add_column(
        "sessions",
        sa.Column(
            "chat_values",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "chat_values")
