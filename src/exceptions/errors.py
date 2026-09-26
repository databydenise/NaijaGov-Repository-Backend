"""
Error factories.

Routes raise these rather than assembling an `HTTPException` by hand, so every error that
leaves this service carries a code the caller can branch on and a message a citizen can
read. Nothing here ever includes a stack trace, a SQL fragment, a header, or a submitted
value.
"""

from typing import Any

from fastapi import HTTPException, status

from src.constants import ErrorCode


def api_error(
    status_code: int,
    code: str,
    message: str,
    **extra: Any,  # noqa: ANN401  # contract extras, e.g. retry_after
) -> HTTPException:
    """An HTTPException whose detail is the error contract's `{code, message}` body."""
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)

    return HTTPException(status_code=status_code, detail=detail)


def unauthenticated() -> HTTPException:
    """
    401, with one message for every cause.

    Missing cookie, expired JWT, wrong signature, wrong `typ`, deleted user, unknown token,
    revoked token, expired token — all identical from outside. Distinguishing them helps
    only someone probing.
    """
    return api_error(
        status.HTTP_401_UNAUTHORIZED,
        ErrorCode.UNAUTHENTICATED,
        "Please sign in again.",
    )


def invalid_credentials() -> HTTPException:
    """401 for a failed login. Deliberately identical for unknown email and wrong password."""
    return api_error(
        status.HTTP_401_UNAUTHORIZED,
        ErrorCode.INVALID_CREDENTIALS,
        "That email and password do not match. Please try again.",
    )


def email_taken() -> HTTPException:
    """
    409 on signup, and specific on purpose.

    Hiding this collision helps nobody: the person already has an account and needs to be
    told to log in instead. It is a different question from a failed login, so it gets a
    different answer.
    """
    return api_error(
        status.HTTP_409_CONFLICT,
        ErrorCode.EMAIL_TAKEN,
        "An account with this email already exists. Please log in instead.",
    )


def not_found(message: str) -> HTTPException:
    """404. Used for another user's row as well as a missing one, never confirming which."""
    return api_error(status.HTTP_404_NOT_FOUND, ErrorCode.NOT_FOUND, message)


def rate_limited(retry_after_seconds: int) -> HTTPException:
    """429 with the wait, so a client can back off instead of hammering."""
    return api_error(
        status.HTTP_429_TOO_MANY_REQUESTS,
        ErrorCode.RATE_LIMITED,
        "Too many attempts. Please wait a moment and try again.",
        retry_after=retry_after_seconds,
    )
