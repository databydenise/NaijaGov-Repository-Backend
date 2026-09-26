"""
The accounts table.

We own identity rather than delegating it to a hosted auth provider, so every user-scoped
foreign key in the schema points here.
"""

from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin, UUIDPk


class User(UUIDPk, TimestampMixin, Base):
    __tablename__ = "users"

    # citext, and normalised to lowercase on the way in. Two guards for one problem:
    # `Ade@x.com` and `ade@x.com` signing up as separate accounts is a support problem.
    email: Mapped[str] = mapped_column(CITEXT, nullable=False, unique=True)

    # argon2id. Never logged, never returned, never in an error message.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
