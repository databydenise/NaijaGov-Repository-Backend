"""Account reads and writes. No route writes SQL; it calls these."""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.users.models import User
from src.users.utils import normalize_email


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """The account for this email, or None. Email is normalised before the lookup."""
    statement = select(User).where(User.email == normalize_email(email))
    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    """The account with this id, or None if it has been deleted."""
    result = await db.execute(select(User).where(User.id == user_id))

    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, password_hash: str) -> User:
    """
    Insert an account and return it.

    Raises `IntegrityError` on a duplicate email; the caller turns that into a 409. We let
    the unique constraint decide rather than checking first, because a check-then-insert
    loses the race between two simultaneous signups.
    """
    user = User(email=normalize_email(email), password_hash=password_hash)
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return user


async def set_password_hash(db: AsyncSession, user_id: uuid.UUID, password_hash: str) -> None:
    """Replace an account's password hash. The caller hashes; nothing raw reaches here."""
    statement = update(User).where(User.id == user_id).values(password_hash=password_hash)

    await db.execute(statement)


async def touch_last_login(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Record a successful login."""
    statement = update(User).where(User.id == user_id).values(last_login_at=func.now())

    await db.execute(statement)
