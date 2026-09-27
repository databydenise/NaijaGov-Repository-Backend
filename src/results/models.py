"""
What `/results` writes: a dedupe marker, a checkpoint event, and durable counters.

Three tables with three different lifetimes, and the differences are the design:

- **`plan_reports`** dies with its session. It exists to make a second report of the same plan
  return the first one's answer instead of writing a second set of rows, so it holds that answer
  and nothing else. A plan lives ten minutes; this row only has to outlive that, and a restart
  must not lose it — an in-process marker would let a retried request double every metric below.
- **`checkpoint_events`** dies with its session too. It is the per-run detail: which kind of
  checkpoint fired, and how far into the batch. Useful while the application is live.
- **`results_counters`** outlives everything. Session rows expire in 24 hours and take their
  action log with them, so without this table "94% of fills stuck" stops being answerable the day
  after. It is deliberately denormalised — text keys, no foreign keys — because a re-seeded
  workflow must not take last week's numbers with it.

No table here has a column for a value, a label, or any page text, and `/results` has no field on
its request that could carry one. This is the endpoint where someone will eventually think it
would be handy to log what was filled; the schema is the answer to that.
"""

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, UUIDPk


class PlanReport(Base):
    """
    One plan, reported once. The row is both the dedupe marker and the replayed answer.

    `plan_id` is the primary key rather than a surrogate, because "one report per plan" is the
    whole point of the table and a unique index on a second column would say it less plainly.

    `response` holds the body the first report was answered with, so a retry gets that answer
    back verbatim rather than one recomputed against a session that has since moved to the next
    step. It is counts, codes and one fixed sentence — the same reason P6's explanation cache
    stores its answer instead of its inputs.
    """

    __tablename__ = "plan_reports"

    # A uuid4 hex from `plan.store.new_plan_id()`. Text, not uuid: it is an opaque handle the
    # extension echoes back, and the column's job is to match it, not to parse it.
    plan_id: Mapped[str] = mapped_column(Text, primary_key=True)

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class CheckpointEvent(UUIDPk, Base):
    """
    A run that stopped because the page asked for something only the user can give.

    This is the evidence for the safety claim, and the way you would notice a detector that never
    fires. `reason` is a short kind — "password", "otp" — never the content of anything, and
    `after_index` is how many actions had already run, which is a position in a batch rather than
    anything about the page.
    """

    __tablename__ = "checkpoint_events"
    __table_args__ = (
        CheckConstraint("after_index >= 0", name="after_index_non_negative"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The plan the run belonged to. Not a foreign key to `plan_reports`: the event is written in
    # the same transaction as that row, and a constraint between two rows written together buys
    # nothing but an ordering requirement.
    plan_id: Mapped[str] = mapped_column(Text, nullable=False)

    reason: Mapped[str] = mapped_column(Text, nullable=False)
    after_index: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ResultsCounter(UUIDPk, Base):
    """
    One counter per workflow, step, metric and day. The only thing here that outlives a session.

    `metric` is wider than the spec's "status" on purpose, because the spec's own list of
    questions needs it to be: fill success rate per step is a status count, but *"how often each
    checkpoint kind fires"* is not a status at all. So one column holds either — a bare status
    (`ok`, `changed`, …), `checkpoint:<kind>`, or `abort:<reason>` — and one upsert path serves
    all three. A second table per metric family would have been three code paths answering one
    question each.

    `workflow_id` and `step_id` are text with a sentinel for "unknown", not nullable foreign
    keys, and both choices are deliberate. No foreign key, because a re-seeded or retired
    workflow must not delete or blank the numbers it produced. No nulls, because Postgres treats
    them as distinct in a unique index, which would quietly break the upsert for every page
    outside the registry — one new row per report instead of one incremented row.
    """

    __tablename__ = "results_counters"
    __table_args__ = (
        Index(
            "uq_results_counters_key",
            "workflow_id",
            "step_id",
            "metric",
            "day",
            unique=True,
        ),
    )

    workflow_id: Mapped[str] = mapped_column(Text, nullable=False)
    step_id: Mapped[str] = mapped_column(Text, nullable=False)
    metric: Mapped[str] = mapped_column(Text, nullable=False)

    # UTC, so a day does not shift with a deployment — the same reason `agent/quota.py` keys on a
    # UTC date.
    day: Mapped[date] = mapped_column(Date, nullable=False)

    count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
