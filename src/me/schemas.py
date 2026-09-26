"""`/me` response models."""

from datetime import datetime

from pydantic import BaseModel

from src.auth.schemas import UserOut
from src.profiles.schemas import ProfileOut


class TokenSummary(BaseModel):
    """
    Enough about the caller's token to warn before it expires. No hash, no raw value.

    `expires_at` lets the panel say "reconnect soon" before the token dies mid-demo.
    """

    expires_at: datetime
    last4: str | None


class MeResponse(BaseModel):
    """
    Everything the panel needs to leave IDLE, in one round trip.

    The extension caches `profile` against `profile_version`; this response wins on any
    conflict with that cache.
    """

    user: UserOut
    profile: ProfileOut
    profile_version: int
    token: TokenSummary
    supported_hosts: list[str]
