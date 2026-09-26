"""
Profile endpoint. Web app only, on `require_user`.

Read-only by decision: there is no write path, so there is also no consent screen or
delete flow to build yet. Values a user gives in chat live in the session row and expire
with it; they are never written here.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.dependencies import require_user
from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.profiles import service as profiles_service
from src.profiles.schemas import ProfileResponse

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=ProfileResponse)
async def read_profile(
    user: Annotated[AuthedUser, Depends(require_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ProfileResponse:
    """The signed-in user's profile. Never a 404: `require_user` guarantees the row."""
    return await profiles_service.read_profile(db, user.id)
