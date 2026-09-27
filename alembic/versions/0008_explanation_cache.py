"""
The explanation cache: `/explain`'s answers, shared across users.

A table rather than a dict in the process, for one specific reason: the whole value of this cache
is a second pass over a form feeling instant, and `fastapi dev` restarts on every file save — so an
in-process cache would be empty between a rehearsal and the run that matters.

Revision ID: 0008_explanation_cache
Revises: 0007_sessions_chat_values
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0008_explanation_cache"
down_revision: str | None = "0007_sessions_chat_values"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

KEY_INDEX = "uq_explanation_cache_cache_key"
EXPIRY_INDEX = "ix_explanation_cache_expires_at"


def upgrade() -> None:
    op.create_table(
        "explanation_cache",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        # SHA-256 of the key's parts: workflow, step, normalised label, prompt version, and the
        # agency's ingestion timestamp. Hashed rather than composite because one part is a page
        # label, and a label on a government form can name a medical condition.
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        # Null where an example would be meaningless, which is most labels.
        sa.Column("example", sa.Text(), nullable=True),
        # The citations as the panel shows them, already verified when the row was written.
        sa.Column(
            "sources",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # A week. Long enough that a demo is fast, short enough that a corpus update cannot leave
        # a stale answer in front of a citizen indefinitely. Invalidation is otherwise by
        # construction: re-ingesting an agency's material changes the key, so the old row is
        # simply never looked up again.
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_explanation_cache"),
    )

    # The lookup, and the conflict target for the write: two panels asking about the same field
    # at the same moment leave one row rather than two.
    op.create_index(KEY_INDEX, "explanation_cache", ["cache_key"], unique=True)

    # For the cleanup sweep. Reads filter on `expires_at` regardless.
    op.create_index(EXPIRY_INDEX, "explanation_cache", ["expires_at"])

    # Defence in depth, as on every other table in this schema. No policy is written, so a future
    # read-only role sees zero rows until someone writes one deliberately.
    op.execute("ALTER TABLE explanation_cache ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.drop_index(EXPIRY_INDEX, table_name="explanation_cache")
    op.drop_index(KEY_INDEX, table_name="explanation_cache")
    op.drop_table("explanation_cache")
