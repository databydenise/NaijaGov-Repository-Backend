"""Extension token lookups.

Callers pass a hash, never a raw token, and nothing here returns or logs a token value.
"""

import uuid
from datetime import datetime

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


async def get_token_by_id(db: AsyncSession, token_id: uuid.UUID) -> ExtensionToken | None:
    """
    The token row with this id, in any state. Never use this to authenticate.

    For reading a token `require_token` has already verified in this session: `db.get`
    answers from the identity map, so `/me` gets `expires_at` and `last4` without a query.
    """
    return await db.get(ExtensionToken, token_id)


async def touch_token(db: AsyncSession, token_id: uuid.UUID) -> None:
    """Record that a token was used just now."""
    statement = (
        update(ExtensionToken)
        .where(ExtensionToken.id == token_id)
        .values(last_used_at=func.now())
    )

    await db.execute(statement)


async def get_active_token(db: AsyncSession, user_id: uuid.UUID) -> ExtensionToken | None:
    """
    The user's one usable token, or None.

    Filters the expiry as well as revocation, which the partial unique index cannot: `now()`
    is not immutable and so cannot appear in an index predicate.
    """
    statement = select(ExtensionToken).where(
        ExtensionToken.user_id == user_id,
        ExtensionToken.revoked_at.is_(None),
        ExtensionToken.expires_at > func.now(),
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def revoke_active_tokens(db: AsyncSession, user_id: uuid.UUID) -> int:
    """
    Revoke every unrevoked token this user has. Returns how many were revoked.

    Runs before issuing a new one, both because one active token per user is the product
    decision and because the partial unique index would otherwise reject the insert.
    """
    statement = (
        update(ExtensionToken)
        .where(
            ExtensionToken.user_id == user_id,
            ExtensionToken.revoked_at.is_(None),
        )
        .values(revoked_at=func.now())
    )

    result = await db.execute(statement)

    return result.rowcount


async def create_token(
    db: AsyncSession,
    user_id: uuid.UUID,
    token_hash: str,
    last4: str,
    label: str | None,
    expires_at: datetime,
) -> ExtensionToken:
    """
    Store a new token. The caller passes a hash; the raw value never reaches this layer.
    """
    token = ExtensionToken(
        user_id=user_id,
        token_hash=token_hash,
        last4=last4,
        label=label,
        expires_at=expires_at,
    )

    db.add(token)
    await db.flush()
    await db.refresh(token)

    return token


async def get_token_row_by_hash_any_state(
    db: AsyncSession,
    token_hash: str,
) -> ExtensionToken | None:
    """
    The row for this hash whether or not it is revoked or expired.

    **Never use this to authenticate.** It exists so demo setup can find a configured demo
    token that was revoked earlier and restore it, instead of inserting a duplicate hash that
    the unique constraint would reject. Authentication goes through `get_token_by_hash`.
    """
    result = await db.execute(
        select(ExtensionToken).where(ExtensionToken.token_hash == token_hash),
    )

    return result.scalar_one_or_none()


async def reactivate_token(
    db: AsyncSession,
    token_id: uuid.UUID,
    expires_at: datetime,
) -> None:
    """
    Clear a token's revocation and give it a new expiry.

    Used only by demo setup. The caller must revoke the user's other active tokens first, or
    the one-active-token index rejects this.
    """
    statement = (
        update(ExtensionToken)
        .where(ExtensionToken.id == token_id)
        .values(revoked_at=None, expires_at=expires_at)
    )

    await db.execute(statement)


async def revoke_token_by_id(
    db: AsyncSession,
    user_id: uuid.UUID,
    token_id: uuid.UUID,
) -> bool:
    """
    Revoke one token belonging to this user. False if it is not theirs or does not exist.

    Scoping the UPDATE by `user_id` is what makes another user's token indistinguishable
    from a token that was never there — the caller returns the same 404 for both and so
    never confirms that someone else's row exists.
    """
    statement = (
        update(ExtensionToken)
        .where(
            ExtensionToken.id == token_id,
            ExtensionToken.user_id == user_id,
            ExtensionToken.revoked_at.is_(None),
        )
        .values(revoked_at=func.now())
    )

    result = await db.execute(statement)

    return result.rowcount > 0
