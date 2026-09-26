"""
Signup and login logic, and the cookie itself.

Handlers parse and return; the decisions live here.
"""

import logging
import uuid

from fastapi import Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.constants import SESSION_COOKIE_NAME
from src.auth.schemas import UserOut
from src.auth.security import (
    DUMMY_PASSWORD_HASH,
    create_session_jwt,
    hash_password,
    verify_password,
)
from src.config import settings
from src.exceptions.errors import email_taken, invalid_credentials
from src.profiles import service as profiles_service
from src.users import service as users_service

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


def set_session_cookie(response: Response, user_id: uuid.UUID) -> None:
    """
    Attach the session cookie.

    httpOnly so script cannot read it; Secure because Chrome treats `http://localhost` as a
    secure context, so this works in local development too; SameSite=None because the web
    app and this API are on different origins, which also forces Secure.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_jwt(user_id),
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=settings.session_ttl_days * SECONDS_PER_DAY,
    )


def clear_session_cookie(response: Response) -> None:
    """
    Remove the session cookie.

    The attributes must match the ones it was set with, or the browser keeps the original.
    Note this ends the session for the user but does not invalidate the JWT, which stays
    valid until it expires — deliberate, and recorded as work to do before real users.
    """
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=True,
        samesite="none",
    )


async def signup(db: AsyncSession, response: Response, email: str, password: str) -> UserOut:
    """
    Create an account, give it a profile, and sign it in.

    A duplicate email is caught from the unique constraint rather than checked for first, so
    two simultaneous signups cannot both succeed.
    """
    try:
        user = await users_service.create_user(db, email, hash_password(password))
    except IntegrityError as exc:
        await db.rollback()

        raise email_taken() from exc

    await profiles_service.ensure_profile(db, user.id)
    set_session_cookie(response, user.id)

    logger.info("Account created: %s", user.id)

    return UserOut(id=user.id, email=user.email)


async def login(db: AsyncSession, response: Response, email: str, password: str) -> UserOut:
    """
    Verify credentials and sign in.

    An unknown email still pays for one argon2 verification against a dummy hash, so a
    missing account and a wrong password take the same time and return the same error.
    """
    user = await users_service.get_user_by_email(db, email)

    if user is None:
        verify_password(password, DUMMY_PASSWORD_HASH)

        raise invalid_credentials()

    if not verify_password(password, user.password_hash):
        raise invalid_credentials()

    await users_service.touch_last_login(db, user.id)
    set_session_cookie(response, user.id)

    logger.info("Login succeeded: %s", user.id)

    return UserOut(id=user.id, email=user.email)
