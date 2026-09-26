"""
Extension tokens gain last4 and a one-active-token index.

Revision ID: 0003_token_last4
Revises: 0002_users_auth
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_token_last4"
down_revision: str | None = "0002_users_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ONE_ACTIVE_INDEX = "uq_extension_tokens_one_active"


def upgrade() -> None:
    # The raw token is unrecoverable by design, so the profile page needs something it can
    # display to identify which token it is looking at.
    op.add_column("extension_tokens", sa.Column("last4", sa.Text(), nullable=True))

    # One active token per user. The predicate omits the expiry on purpose: `now()` is not
    # immutable and Postgres will not accept it in an index predicate. An expired token
    # therefore still occupies the slot until it is revoked or replaced, which is exactly
    # what `POST /tokens` does before issuing a new one.
    op.create_index(
        ONE_ACTIVE_INDEX,
        "extension_tokens",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(ONE_ACTIVE_INDEX, table_name="extension_tokens")
    op.drop_column("extension_tokens", "last4")
