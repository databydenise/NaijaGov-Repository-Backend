"""
The explanation cache: one row per answer, shared by every user.

Shared is the point. "What is Business Type?" has the same answer for everyone on the same step of
the same workflow, so the row is keyed by the *question's shape* — workflow, step, normalised
label, prompt version, corpus vintage — and never by who asked. That is what makes the second pass
over a six-field form instant.

Sharing a table between users is also why two things are absent from it:

- **No user column, and nothing user-specific in a row.** An explanation describes a public form
  field and cites public sources. If anything user-specific could reach this table, the table
  would be the wrong design rather than a row needing redaction.
- **No question text.** A custom question is answered and never cached (`store.py` builds no key
  for one), so a citizen's own words about their own application do not end up in a row every
  other account can be served from.

An ungrounded answer is never written here either. Those are the ones worth retrying once the
corpus grows, and a week-long row would make the retry impossible.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, UUIDPk


class ExplanationCache(UUIDPk, Base):
    __tablename__ = "explanation_cache"
    __table_args__ = (
        # The lookup, and the conflict target for the write. Unique, so two panels asking about
        # the same field at the same moment leave one row rather than two.
        Index("uq_explanation_cache_cache_key", "cache_key", unique=True),
        # For the cleanup sweep. Reads filter on `expires_at` regardless, so a sweep that never
        # runs costs storage and never a stale answer.
        Index("ix_explanation_cache_expires_at", "expires_at"),
    )

    # SHA-256 of the key's parts. Hashed rather than stored as a readable composite because one
    # of its parts is a page label, and a label on a government form can name a medical
    # condition or a benefit — not something to leave lying in an index on a shared table.
    cache_key: Mapped[str] = mapped_column(Text, nullable=False)

    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    # Null where an example would be meaningless. The column is nullable for the same reason the
    # model's field is: "Declaration" has no example value, and inventing one is the failure this
    # project exists to prevent.
    example: Mapped[str | None] = mapped_column(Text)

    # `[{"title": ..., "url": ..., "checked": "2026-09-20"}]` — the citations as the panel shows
    # them, already verified against the chunks the answer was grounded in. Stored rather than
    # re-derived, because the chunk a row cited may be gone by the time the row is read.
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )

    # Part of the key as well as a column. Kept as a column so a row can be read and explained
    # by a person looking at the table, which a hash cannot be.
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
