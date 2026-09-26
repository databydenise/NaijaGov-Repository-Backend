"""Extension token lookups.

Callers pass a hash, never a raw token, and nothing here returns or logs a token value.
"""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.tokens.models import ExtensionToken


async def get_token_by_hash(db: AsyncSession, token_hash: str) -> ExtensionToken | None:
    """An active token matching this hash, or None.

    Revoked and expired tokens are filtered here rather than by the caller, so no route
    can forget to check. A None result is an `UNAUTHENTICATED` response, and the caller
    is not told which of the three reasons applied.
    """
    statement = select(ExtensionToken).where(
        ExtensionToken.token_hash == token_hash,
        ExtensionToken.revoked_at.is_(None),
        ExtensionToken.expires_at > func.now(),
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def touch_token(db: AsyncSession, token_id: uuid.UUID) -> None:
    """Record that a token was used just now."""
    statement = (
        update(ExtensionToken)
        .where(ExtensionToken.id == token_id)
        .values(last_used_at=func.now())
    )

    await db.execute(statement)
