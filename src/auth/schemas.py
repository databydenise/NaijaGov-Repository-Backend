"""Auth request and response models."""

import uuid

from pydantic import BaseModel, EmailStr, Field

from src.auth.constants import PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH


class SignupRequest(BaseModel):
    """Email and password. Nothing else is collected at signup."""

    email: EmailStr
    password: str = Field(
        min_length=PASSWORD_MIN_LENGTH,
        max_length=PASSWORD_MAX_LENGTH,
    )


class LoginRequest(BaseModel):
    """
    Credentials for a login attempt.

    No length bounds on the password here: rejecting a 200-character password with a
    validation error would tell an attacker their guess was the wrong *shape*, and the
    comparison against a stored hash costs the same either way. The cap that matters is on
    signup, where the input is what gets hashed.
    """

    email: EmailStr
    password: str = Field(max_length=1024)


class UserOut(BaseModel):
    """The public view of an account. Never carries `password_hash`."""

    id: uuid.UUID
    email: str


class AuthResponse(BaseModel):
    """What signup and login return. The session itself travels as a cookie."""

    user: UserOut


class SessionResponse(BaseModel):
    """What the web app reads on load to decide whether to show the signed-in area."""

    user: UserOut
    profile_version: int


class AuthedUser(BaseModel):
    """
    The caller, as resolved by `require_user` or `require_token`.

    Both dependencies return this same shape so a route body does not need to know which
    credential got it here — but no route ever takes both dependencies, so in practice a
    route knows exactly one of them could have run.
    """

    id: uuid.UUID
    email: str

    # Populated by `require_user`, which calls `ensure_profile` anyway. `require_token`
    # leaves it unset rather than paying for a second query on the extension's hottest path.
    profile_version: int | None = None

    # Populated by `require_token` only. The token's id, never its value.
    token_id: uuid.UUID | None = None
