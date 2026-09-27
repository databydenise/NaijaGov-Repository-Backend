"""
What `/results` needs: wider statuses, session counters, and three tables.

Four changes, in the order a reader would want them:

1. `ck_action_log_status` is widened. `changed` and `cancelled` are what the content script can
   report and the guard never could, so until now there was no status for "the portal rewrote the
   value" or "the batch stopped before this one ran".
2. `sessions` gains one counter per status — this application's own running tally, which dies
   with the session.
3. `plan_reports` makes reporting idempotent, durably. An in-process marker would be lost on a
   reload and a retried report would then double every counter below.
4. `checkpoint_events` and `results_counters` are the evidence and the metrics. The counters are
   the only thing here that outlives a session, which is what makes "94% of fills stuck" still
   answerable next week.

Revision ID: 0009_results
Revises: 0008_explanation_cache
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0009_results"
down_revision: str | None = "0008_explanation_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATUS_CHECK = "ck_action_log_status"

# Literal copies, not imports of `sessions/constants.py`. A migration has to keep describing the
# schema it created even after the application constant moves on — the same reason
# `0006_documents` spells out its embedding width instead of importing it.
OLD_ACTION_STATUSES = ("ok", "failed", "rejected")
NEW_ACTION_STATUSES = ("ok", "changed", "failed", "rejected", "cancelled")

# One column per status, all defaulting to zero.
COUNTER_COLUMNS = (
    "results_ok",
    "results_changed",
    "results_failed",
    "results_rejected",
    "results_cancelled",
)

COUNTERS_KEY_INDEX = "uq_results_counters_key"


def _in_clause(column: str, values: tuple[str, ...]) -> str:
    """Render `column IN ('a', 'b')` from a code-owned tuple of literals."""
    rendered = ", ".join(f"'{value}'" for value in values)

    return f"{column} IN ({rendered})"


def upgrade() -> None:
    # 1 — the wider status vocabulary. Text plus CHECK rather than an enum is exactly so this is
    # a two-line migration; `0001_initial` says so where it created the constraint.
    op.drop_constraint(STATUS_CHECK, "action_log", type_="check")
    op.create_check_constraint(
        STATUS_CHECK,
        "action_log",
        _in_clause("status", NEW_ACTION_STATUSES),
    )

    # 2 — the session's own tally. `server_default` rather than a backfill: every existing row
    # should read zero, which is what it has actually recorded.
    for column in COUNTER_COLUMNS:
        op.add_column(
            "sessions",
            sa.Column(column, sa.Integer(), server_default=sa.text("0"), nullable=False),
        )

    # 3 — one report per plan, and the answer it was given.
    op.create_table(
        "plan_reports",
        # The plan id itself, not a surrogate: "one report per plan" is the table's whole purpose,
        # and a primary key says that more plainly than a unique index on a second column.
        sa.Column("plan_id", sa.Text(), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        # The response the first report was answered with, replayed verbatim on a retry rather
        # than recomputed against a session that may have moved to the next step since.
        sa.Column("response", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_plan_reports_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("plan_id", name="pk_plan_reports"),
    )
    op.create_index("ix_plan_reports_session_id", "plan_reports", ["session_id"])

    # 4a — the per-run detail of a stop. Cascades with its session: this is what someone reads
    # while an application is live, and the durable version of it is a counter.
    op.create_table(
        "checkpoint_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plan_id", sa.Text(), nullable=False),
        # A short kind — "password", "otp" — never the content of anything.
        sa.Column("reason", sa.Text(), nullable=False),
        # How many actions had already run. A position in a batch, not a fact about the page.
        sa.Column("after_index", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_checkpoint_events_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_checkpoint_events"),
        sa.CheckConstraint(
            "after_index >= 0",
            name="ck_checkpoint_events_after_index_non_negative",
        ),
    )
    op.create_index("ix_checkpoint_events_session_id", "checkpoint_events", ["session_id"])

    # 4b — the durable counters. No foreign keys on purpose: a re-seeded or retired workflow must
    # not delete or blank the numbers it produced, so the keys are text snapshots rather than
    # references. NOT NULL with a '-' sentinel for an unknown workflow or step, because Postgres
    # treats NULLs as distinct in a unique index and the upsert would insert instead of increment.
    op.create_table(
        "results_counters",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("workflow_id", sa.Text(), nullable=False),
        sa.Column("step_id", sa.Text(), nullable=False),
        # A result status, or `checkpoint:<kind>`, or `abort:<reason>`. Wider than the spec's
        # "status" because the spec's own list of questions includes checkpoint frequency, which
        # is not a status.
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_results_counters"),
    )
    op.create_index(
        COUNTERS_KEY_INDEX,
        "results_counters",
        ["workflow_id", "step_id", "metric", "day"],
        unique=True,
    )

    # Defence in depth, as on every other table in this schema. No policy is written, so a future
    # read-only role sees zero rows until someone writes one deliberately.
    for table in ("plan_reports", "checkpoint_events", "results_counters"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    """
    Reverse of upgrade. A migration that cannot be reversed cannot be tested twice.

    The status CHECK goes back to its three values, which means this only reverses cleanly while
    no `changed` or `cancelled` row exists — and failing loudly on one is right: silently deleting
    a citizen's run history to narrow a constraint is not a downgrade anyone asked for.
    """
    op.drop_index(COUNTERS_KEY_INDEX, table_name="results_counters")
    op.drop_table("results_counters")

    op.drop_index("ix_checkpoint_events_session_id", table_name="checkpoint_events")
    op.drop_table("checkpoint_events")

    op.drop_index("ix_plan_reports_session_id", table_name="plan_reports")
    op.drop_table("plan_reports")

    for column in COUNTER_COLUMNS:
        op.drop_column("sessions", column)

    op.drop_constraint(STATUS_CHECK, "action_log", type_="check")
    op.create_check_constraint(
        STATUS_CHECK,
        "action_log",
        _in_clause("status", OLD_ACTION_STATUSES),
    )
