"""Initial schema: accounts, profiles, tokens, workflow registry, sessions, action log.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-25

Hand-written rather than left as autogenerate produced it. Autogenerate does not know about
the pgcrypto extension, the `set_updated_at()` trigger, or row-level security, and would
have silently omitted all three.

Identity note: `profiles`, `extension_tokens`, and `sessions` reference `public.users`.
This service owns the accounts table.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables carrying `updated_at`, and therefore an UPDATE trigger. `extension_tokens` and
# `action_log` are absent on purpose: every change they get is an explicit column.
TRIGGER_TABLES: tuple[str, ...] = (
    "users",
    "profiles",
    "workflows",
    "workflow_steps",
    "rules",
    "sessions",
)

# Every table gets RLS, with no policies. This service connects as the table owner and so
# bypasses it entirely; any other role gets nothing. It is defence in depth against a
# second, less privileged connection appearing later — and the reason to know about it is
# that a future read-only role will see zero rows until a policy is written for it.
ALL_TABLES: tuple[str, ...] = (
    "users",
    "profiles",
    "extension_tokens",
    "workflows",
    "workflow_steps",
    "rules",
    "sessions",
    "action_log",
)

ACTION_TYPES = (
    "fill",
    "select",
    "check",
    "highlight",
    "scroll",
    "explain",
    "clickSafe",
    "pause",
)
ACTION_STATUSES = ("ok", "failed", "rejected")


def _in_clause(column: str, values: tuple[str, ...]) -> str:
    rendered = ", ".join(f"'{value}'" for value in values)

    return f"{column} IN ({rendered})"


def upgrade() -> None:
    # 1 — gen_random_uuid() lives in pgcrypto.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # 2 — the trigger function. Triggers themselves are attached after their tables exist.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS trigger AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
    )

    # 3 — tables, in foreign-key order.
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "profiles",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("full_name", sa.Text(), nullable=True),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("phone", sa.Text(), nullable=True),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=True),
        sa.Column("lga", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_profiles_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_profiles"),
    )

    op.create_table(
        "extension_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_extension_tokens_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_extension_tokens"),
        # /me looks a token up on every call: one index hit, and the uniqueness is the
        # guarantee that two tokens cannot collide.
        sa.UniqueConstraint("token_hash", name="uq_extension_tokens_token_hash"),
    )
    op.create_index(
        "ix_extension_tokens_user_id",
        "extension_tokens",
        ["user_id"],
    )

    op.create_table(
        "workflows",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("agency", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "url_patterns",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflows"),
    )

    op.create_table(
        "workflow_steps",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workflow_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column(
            "field_labels",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "is_final",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
            name="fk_workflow_steps_workflow_id_workflows",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_steps"),
        sa.UniqueConstraint(
            "workflow_id",
            "index",
            name="uq_workflow_steps_workflow_index",
        ),
    )
    op.create_index(
        "ix_workflow_steps_workflow_id",
        "workflow_steps",
        ["workflow_id"],
    )

    op.create_table(
        "rules",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workflow_id", sa.Text(), nullable=False),
        sa.Column("step_id", sa.Text(), nullable=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("field_label", sa.Text(), nullable=True),
        sa.Column("requirement", sa.Text(), nullable=False),
        # NOT NULL is what makes "no rule, no claim" true in the data and not only in
        # the prompt.
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("last_checked", sa.Date(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
            name="fk_rules_workflow_id_workflows",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["step_id"],
            ["workflow_steps.id"],
            name="fk_rules_step_id_workflow_steps",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_rules"),
    )

    op.create_table(
        "sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("workflow_id", sa.Text(), nullable=True),
        sa.Column("step_id", sa.Text(), nullable=True),
        sa.Column("page_hash", sa.Text(), nullable=True),
        sa.Column(
            "cached_rule_ids",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "history",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_sessions_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_id"],
            ["workflows.id"],
            name="fk_sessions_workflow_id_workflows",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["step_id"],
            ["workflow_steps.id"],
            name="fk_sessions_step_id_workflow_steps",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
    )

    op.create_table(
        "action_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        # The field's id in the page snapshot. There is no value column, and adding one
        # is a conversation, not a migration.
        sa.Column("field_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_action_log_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_action_log"),
        # Text plus CHECK, not an enum: this list mirrors the extension's action union and
        # will change during the build. Altering a CHECK is a one-line migration.
        sa.CheckConstraint(
            _in_clause("action_type", ACTION_TYPES),
            name="ck_action_log_action_type",
        ),
        sa.CheckConstraint(
            _in_clause("status", ACTION_STATUSES),
            name="ck_action_log_status",
        ),
    )
    op.create_index("ix_action_log_session_id", "action_log", ["session_id"])

    # 4 — the remaining indexes: the two lookups B9 and /explain make, and the two
    # session reads (newest for a user, and the expiry sweep).
    op.create_index("ix_rules_workflow_step", "rules", ["workflow_id", "step_id"])
    op.create_index(
        "ix_rules_workflow_field_label",
        "rules",
        ["workflow_id", "field_label"],
    )
    op.create_index(
        "ix_sessions_user_updated_at",
        "sessions",
        ["user_id", sa.text("updated_at DESC")],
    )
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])

    # 5 — triggers, now that the tables exist.
    for table in TRIGGER_TABLES:
        op.execute(
            f"""
            CREATE TRIGGER set_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION set_updated_at()
            """,  # noqa: S608  # table names come from a module constant
        )

    # 6 — row-level security on every table, with no policies created.
    for table in ALL_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")  # noqa: S608


def downgrade() -> None:
    """Reverse of upgrade. A migration that cannot be reversed cannot be tested twice."""
    for table in TRIGGER_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS set_{table}_updated_at ON {table}")  # noqa: S608

    op.drop_index("ix_sessions_expires_at", table_name="sessions")
    op.drop_index("ix_sessions_user_updated_at", table_name="sessions")
    op.drop_index("ix_rules_workflow_field_label", table_name="rules")
    op.drop_index("ix_rules_workflow_step", table_name="rules")
    op.drop_index("ix_action_log_session_id", table_name="action_log")
    op.drop_index("ix_workflow_steps_workflow_id", table_name="workflow_steps")
    op.drop_index("ix_extension_tokens_user_id", table_name="extension_tokens")

    op.drop_table("action_log")
    op.drop_table("sessions")
    op.drop_table("rules")
    op.drop_table("workflow_steps")
    op.drop_table("workflows")
    op.drop_table("extension_tokens")
    op.drop_table("profiles")
    op.drop_table("users")

    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

    # pgcrypto is left installed: other schemas in the same database may rely on it, and
    # dropping an extension is not this migration's business.
