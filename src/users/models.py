"""The accounts table.

We own identity rather than delegating it to a hosted auth provider, so every user-scoped
foreign key in the schema points here.
"""

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin, UUIDPk


class User(UUIDPk, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # argon2id. Never logged, never returned, never in an error message. Written for the
    # first time in B3 — this task only defines the column.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
