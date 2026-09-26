"""
Users table gains citext email and last_login_at.

Revision ID: 0002_users_auth
Revises: 0001_initial
Create Date: 2026-09-26

B2 created `users` with a plain `text` email. This adds only what auth needs: the citext
extension, the column type change, and a login timestamp. The foreign keys the auth spec
asks for — `profiles.user_id` and `extension_tokens.user_id` cascading to `users(id)` —
already exist from `0001_initial`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_users_auth"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    # The unique index on email is rebuilt by Postgres as part of the type change, so the
    # uniqueness guarantee is never dropped — it becomes case-insensitive.
    op.execute("ALTER TABLE users ALTER COLUMN email TYPE citext")

    op.add_column(
        "users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "last_login_at")
    op.execute("ALTER TABLE users ALTER COLUMN email TYPE text")

    # citext is left installed. Dropping an extension other objects may use is not this
    # migration's business, and re-creating it is free.
