"""
Extension token generation. Pure functions, no database, no logging.

A token grants form-filling rights on a government portal. If one leaks, the answer has to
be short: it is 32 random bytes, it is stored only as a SHA-256, it expires, and it can be
revoked in one request.
"""

import hashlib
import hmac
import secrets

# Makes a leaked token greppable in a scan and unmistakable in a support screenshot.
TOKEN_PREFIX = "ngv_"

TOKEN_BYTES = 32
LAST4_LENGTH = 4

MAX_LABEL_LENGTH = 60


def hash_token(raw: str) -> str:
    """
    SHA-256 of a raw token, hex encoded.

    Not argon2: a token is already high-entropy, and `/me` looks one up on every panel
    open. A slow KDF on that path buys nothing and costs latency.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_token() -> tuple[str, str, str]:
    """
    A new token as `(raw, token_hash, last4)`.

    The raw value is returned to exactly one caller, included in exactly one response, and
    never stored, logged, or recoverable afterwards.
    """
    raw = f"{TOKEN_PREFIX}{secrets.token_urlsafe(TOKEN_BYTES)}"

    return raw, hash_token(raw), raw[-LAST4_LENGTH:]


def verify_token(raw: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented token against a stored hash."""
    return hmac.compare_digest(hash_token(raw), stored_hash)


def has_token_prefix(raw: str) -> bool:
    """Cheap shape check, so a junk credential is rejected before the database is touched."""
    return raw.startswith(TOKEN_PREFIX)
