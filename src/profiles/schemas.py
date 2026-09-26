"""Profile response models. There is no request model: the profile is read-only for now."""

from datetime import datetime

from pydantic import BaseModel


class ProfileOut(BaseModel):
    """
    The fillable profile. Every field may be null, and a null is a normal state.

    Shared by `GET /profile` and `GET /me`, so the web app and the extension can never
    disagree about what a profile looks like.
    """

    full_name: str | None
    email: str | None
    phone: str | None
    address: str | None
    state: str | None
    lga: str | None


class Completeness(BaseModel):
    """How much of the profile is filled. Computed on read, never stored."""

    filled: int
    total: int
    missing: list[str]


class ProfileResponse(BaseModel):
    """`GET /profile`."""

    profile: ProfileOut
    version: int
    updated_at: datetime
    completeness: Completeness
