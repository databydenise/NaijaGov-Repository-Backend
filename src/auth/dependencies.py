"""
`require_user` — the web app's session cookie.

This dependency and `require_token` are never listed on the same route. A browser session
must not be able to drive a form fill, and an extension token must not be able to change
account settings. That separation is the whole point of having two.
"""

from typing import Annotated

from fastapi import Cookie, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.constants import SESSION_COOKIE_NAME
from src.auth.schemas import AuthedUser
from src.auth.security import decode_session_jwt
from src.database.session import get_session
from src.exceptions.errors import unauthenticated
from src.profiles import service as profiles_service
from src.users import service as users_service


async def require_user(
    db: Annotated[AsyncSession, Depends(get_session)],
    ngv_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> AuthedUser:
    """
    The signed-in account, or a 401.

    Every failure — no cookie, expired, wrong signature, wrong `typ`, user since deleted —
    raises the same error with the same message. A profile row is created if the user does
    not have one, so nothing downstream has to cope with its absence.
    """
    if not ngv_session:
        raise unauthenticated()

    user_id = decode_session_jwt(ngv_session)

    if user_id is None:
        raise unauthenticated()

    user = await users_service.get_user_by_id(db, user_id)

    if user is None:
        raise unauthenticated()

    profile = await profiles_service.ensure_profile(db, user.id)

    return AuthedUser(
        id=user.id,
        email=user.email,
        profile_version=profile.version,
    )
