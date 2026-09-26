"""Extension tokens.

The raw token is 32 bytes of CSPRNG, shown to the user exactly once, and never stored or
logged. Only its SHA-256 lands here. `/me` looks a token up on every call, so `token_hash`
carries a unique index and that lookup is a single index hit.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, UUIDPk


class ExtensionToken(UUIDPk, Base):
    __tablename__ = "extension_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # Human label, e.g. "Work laptop". Set when the web app issues the token.
    label: Mapped[str | None] = mapped_column(Text)

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # No `updated_at`: every mutation this row gets is an explicit column
    # (`revoked_at`, `last_used_at`), so there is nothing for a trigger to say.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
