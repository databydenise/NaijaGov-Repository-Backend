"""
`GET /me`. Extension only, on `require_token`.

The web app's equivalent is `GET /auth/session` on the session cookie. A cookie presented
here is ignored and the call is a 401, and a bearer token presented there is the same.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.me import service as me_service
from src.me.schemas import MeResponse
from src.tokens.dependencies import require_token

router = APIRouter(prefix="/me", tags=["extension"])


@router.get("", response_model=MeResponse)
async def read_me(
    user: Annotated[AuthedUser, Depends(require_token)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> MeResponse:
    """The extension's identity call, made every time the panel opens."""
    return await me_service.read_me(db, user)
