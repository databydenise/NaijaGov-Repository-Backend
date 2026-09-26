"""
Token endpoints. Web app only, all three on `require_user`.

The extension never calls these — it only carries the result. That is why `require_token`
appears nowhere in this file.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.dependencies import require_user
from src.auth.schemas import AuthedUser
from src.config import settings
from src.database.session import get_session
from src.exceptions.errors import not_found, rate_limited
from src.rate_limit import check_rate_limit
from src.tokens import service as tokens_service
from src.tokens.constants import ISSUE_RATE_LIMIT, ISSUE_RATE_WINDOW_SECONDS
from src.tokens.schemas import IssuedToken, IssueTokenRequest, TokenOut
from src.tokens.utils import generate_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.post("", response_model=IssuedToken, status_code=status.HTTP_201_CREATED)
async def issue_token(
    payload: IssueTokenRequest,
    user: Annotated[AuthedUser, Depends(require_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> IssuedToken:
    """
    Issue a token, revoking whatever the user had before.

    One active token per user: "regenerate" is a clearer story than a list of tokens nobody
    can tell apart. The raw value is in this response and nowhere else, ever.
    """
    limit = check_rate_limit(
        f"tokens:{user.id}",
        ISSUE_RATE_LIMIT,
        ISSUE_RATE_WINDOW_SECONDS,
    )

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    revoked_count = await tokens_service.revoke_active_tokens(db, user.id)
    raw, token_hash, last4 = generate_token()

    token = await tokens_service.create_token(
        db,
        user_id=user.id,
        token_hash=token_hash,
        last4=last4,
        label=payload.label,
        expires_at=datetime.now(UTC) + timedelta(days=settings.token_ttl_days),
    )

    # The id, and only the id. Never the raw value, never the hash.
    logger.info("Token issued: %s (replaced %d)", token.id, revoked_count)

    return IssuedToken(
        id=token.id,
        token=raw,
        last4=last4,
        label=token.label,
        expires_at=token.expires_at,
        replaced_previous=revoked_count > 0,
    )


@router.get("", response_model=list[TokenOut])
async def list_tokens(
    user: Annotated[AuthedUser, Depends(require_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list[TokenOut]:
    """The active token, or an empty list."""
    token = await tokens_service.get_active_token(db, user.id)

    if token is None:
        return []

    return [
        TokenOut(
            id=token.id,
            label=token.label,
            last4=token.last4,
            created_at=token.created_at,
            expires_at=token.expires_at,
            last_used_at=token.last_used_at,
        ),
    ]


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(
    token_id: uuid.UUID,
    user: Annotated[AuthedUser, Depends(require_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """
    Revoke a token. Takes effect on the extension's next call, with no cache to invalidate.

    Someone else's token returns the same 404 as one that never existed, so this never
    confirms that another user's row is there.
    """
    was_revoked = await tokens_service.revoke_token_by_id(db, user.id, token_id)

    if not was_revoked:
        raise not_found("That token could not be found.")

    logger.info("Token revoked: %s", token_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
