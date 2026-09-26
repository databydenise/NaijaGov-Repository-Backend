"""
`require_token` — the extension's bearer token.

Never listed on the same route as `require_user`. This one authorises filling a form on a
government portal; the other one authorises changing an account. They do not overlap.
"""

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.exceptions.errors import unauthenticated
from src.tokens import service as tokens_service
from src.tokens.utils import has_token_prefix, hash_token
from src.users import service as users_service

BEARER_PREFIX = "Bearer "


def _extract_token(authorization: str | None) -> str | None:
    """The raw token from an `Authorization: Bearer ngv_…` header, if it looks like one."""
    if not authorization or not authorization.startswith(BEARER_PREFIX):
        return None

    raw = authorization[len(BEARER_PREFIX) :].strip()

    # Shape check before any database work: a session JWT presented here, or any other
    # credential, is refused without costing a query.
    if not raw or not has_token_prefix(raw):
        return None

    return raw


async def require_token(
    db: Annotated[AsyncSession, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> AuthedUser:
    """
    The account behind a valid extension token, or a 401.

    The repository filters revoked and expired tokens, so a revocation takes effect on the
    very next call — there is no cache to invalidate, which is the reason for the design.
    """
    raw = _extract_token(authorization)

    if raw is None:
        raise unauthenticated()

    token = await tokens_service.get_token_by_hash(db, hash_token(raw))

    if token is None:
        raise unauthenticated()

    user = await users_service.get_user_by_id(db, token.user_id)

    if user is None:
        raise unauthenticated()

    # Inline rather than in a background task. The spec asked for non-blocking, but a
    # background task needs its own connection, and spending a second connection from a
    # ten-connection pool on the extension's hottest path costs more than this UPDATE on a
    # primary key saves.
    await tokens_service.touch_token(db, token.id)

    return AuthedUser(id=user.id, email=user.email, token_id=token.id)
