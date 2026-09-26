"""The profile a fill is drawn from.

Every field is nullable: a user can connect the extension before filling anything in, and
the fill preview's job is to show what is missing. No government ID columns — no NIN, BVN,
passport number, or document upload, here or anywhere.
"""

import uuid

from sqlalchemy import ForeignKey, Integer, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin


class Profile(TimestampMixin, Base):
    __tablename__ = "profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    full_name: Mapped[str | None] = mapped_column(Text)

    # The contact email that goes on forms, not the login email on `users`.
    email: Mapped[str | None] = mapped_column(Text)

    phone: Mapped[str | None] = mapped_column(Text)
    address: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str | None] = mapped_column(Text)
    lga: Mapped[str | None] = mapped_column(Text)

    # Bumped on every write. The extension compares it to decide whether a cached
    # profile is stale.
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
    )
