"""Chat/fill sessions and the log of what the extension actually did.

Two rules hold this file down:

- `sessions.history` is the one place transient personal data lives. It holds the last few
  chat turns and anything the user typed in chat, and it dies with the row.
- `action_log` has **no value column**, and this is the table where one would be most
  tempting. It records which field was touched and how it went, never what was written.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin, UUIDPk
from src.sessions.constants import ACTION_STATUSES, ACTION_TYPES


def _in_clause(column: str, values: tuple[str, ...]) -> str:
    """Render `column IN ('a', 'b')` from a code-owned tuple of literals."""
    rendered = ", ".join(f"'{value}'" for value in values)

    return f"{column} IN ({rendered})"


class Session(UUIDPk, TimestampMixin, Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user_updated_at", "user_id", text("updated_at DESC")),
        Index("ix_sessions_expires_at", "expires_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Null until /context has identified the page. A portal that is retired should not
    # take a user's session history with it, so these clear rather than cascade.
    workflow_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("workflows.id", ondelete="SET NULL"),
    )
    step_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("workflow_steps.id", ondelete="SET NULL"),
    )

    page_hash: Mapped[str | None] = mapped_column(Text)

    # Rule ids already sent to the model for this session, so /plan can skip re-sending
    # them. Ids only — the rule text is looked up fresh.
    cached_rule_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )

    history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )

    # Carried on the row rather than inferred from a cleanup job's schedule: queries
    # filter on it regardless, so a missed sweep is a storage cost, never a wrong answer.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class ActionLog(UUIDPk, Base):
    __tablename__ = "action_log"
    __table_args__ = (
        # Names are completed by the metadata naming convention:
        # ck_action_log_action_type / ck_action_log_status.
        CheckConstraint(_in_clause("action_type", ACTION_TYPES), name="action_type"),
        CheckConstraint(_in_clause("status", ACTION_STATUSES), name="status"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    action_type: Mapped[str] = mapped_column(Text, nullable=False)

    # The field's id in the page snapshot. Not a selector, and not its value.
    field_id: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
