"""
Auth endpoints. Web app only — the extension never calls these.

Note this exposes `GET /auth/session`, not `/me`. `/me` is the extension's endpoint on the
other dependency. Two names, two audiences, no overlap.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import service as auth_service
from src.auth.constants import LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW_SECONDS
from src.auth.dependencies import require_user
from src.auth.schemas import (
    AuthedUser,
    AuthResponse,
    LoginRequest,
    SessionResponse,
    SignupRequest,
    UserOut,
)
from src.database.session import get_session
from src.exceptions.errors import rate_limited
from src.rate_limit import check_rate_limit
from src.users.utils import normalize_email

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    """The caller's address, for rate-limit keying only."""
    return request.client.host if request.client else "unknown"


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> AuthResponse:
    """Create an account and sign it in. A duplicate email returns 409."""
    user = await auth_service.signup(db, response, payload.email, payload.password)

    return AuthResponse(user=user)


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> AuthResponse:
    """
    Sign in with email and password.

    Rate limited on email *and* client IP together, so one attacker cannot spread guesses
    across many accounts from one address, and cannot spread guesses at one account across
    many addresses either.
    """
    key = f"login:{normalize_email(payload.email)}:{_client_ip(request)}"
    limit = check_rate_limit(key, LOGIN_RATE_LIMIT, LOGIN_RATE_WINDOW_SECONDS)

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    user = await auth_service.login(db, response, payload.email, payload.password)

    return AuthResponse(user=user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout() -> Response:
    """
    Clear the session cookie.

    Takes no body and requires no valid session: logging out should never fail. Whoever
    asked to be logged out gets logged out.
    """
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    auth_service.clear_session_cookie(response)

    return response


@router.get("/session", response_model=SessionResponse)
async def read_session(
    user: Annotated[AuthedUser, Depends(require_user)],
) -> SessionResponse:
    """Who is signed in, and which profile version they are on."""
    return SessionResponse(
        user=UserOut(id=user.id, email=user.email),
        profile_version=user.profile_version or 1,
    )
