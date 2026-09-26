"""
Password hashing and session JWTs.

argon2id for passwords because they are low-entropy and need a slow hash; HS256 for the
session cookie because it is our own secret and never leaves this service.
"""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError

from src.auth.constants import JWT_ALGORITHM, JWT_TYPE_SESSION
from src.config import settings

# Defaults, deliberately not hand-tuned.
_hasher = PasswordHasher()

# Compared against on a login for an email that does not exist, so a missing account and a
# wrong password take the same time. Computed once at import, not per request.
DUMMY_PASSWORD_HASH = _hasher.hash("a-password-that-matches-nothing")


def hash_password(raw: str) -> str:
    """argon2id hash of a password. The result is safe to store, never to log."""
    return _hasher.hash(raw)


def verify_password(raw: str, stored: str) -> bool:
    """
    True when the password matches.

    Returns False for a wrong password *and* for a malformed or truncated hash. It never
    raises: a corrupt row in `users` must read as a failed login, not a 500 that tells an
    attacker they found something interesting.
    """
    try:
        return _hasher.verify(stored, raw)
    except (Argon2Error, InvalidHashError, TypeError, ValueError):
        return False


def create_session_jwt(user_id: uuid.UUID) -> str:
    """A signed session token for the cookie, valid for SESSION_TTL_DAYS."""
    issued_at = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": issued_at,
        "exp": issued_at + timedelta(days=settings.session_ttl_days),
        "typ": JWT_TYPE_SESSION,
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_session_jwt(token: str) -> uuid.UUID | None:
    """
    The user id inside a valid session token, or None.

    None covers every failure — bad signature, expired, malformed, wrong `typ`, unparseable
    `sub` — because the caller turns all of them into the same `UNAUTHENTICATED` response.
    Telling them apart would only help someone probing.
    """
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None

    if claims.get("typ") != JWT_TYPE_SESSION:
        return None

    try:
        return uuid.UUID(str(claims.get("sub")))
    except ValueError:
        return None
