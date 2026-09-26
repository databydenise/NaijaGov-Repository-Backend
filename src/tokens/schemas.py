"""Token request and response models."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.tokens.utils import MAX_LABEL_LENGTH


class IssueTokenRequest(BaseModel):
    """An optional human label, so a user can tell one device from another."""

    label: str | None = Field(default=None, max_length=MAX_LABEL_LENGTH)


class IssuedToken(BaseModel):
    """
    The one and only response that carries a raw token.

    It is not stored anywhere in recoverable form and it is never returned again. If the
    user loses it they issue a new one, which revokes this.
    """

    id: uuid.UUID
    token: str
    last4: str
    label: str | None
    expires_at: datetime
    replaced_previous: bool


class TokenOut(BaseModel):
    """
    A token as the profile page sees it. No raw value, no hash.

    `last_used_at` is what lets someone confirm the extension actually connected, which is
    worth more in the interface than it sounds.
    """

    id: uuid.UUID
    label: str | None
    last4: str | None
    created_at: datetime
    expires_at: datetime
    last_used_at: datetime | None
