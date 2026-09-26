"""
Assembling `/me`.

Query budget, on a warm cache: `require_token` spends three (token lookup, user lookup,
`last_used_at` update) and this adds one (the profile). The token row comes from the
session's identity map and the supported hosts from the in-memory cache.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser, UserOut
from src.exceptions.errors import unauthenticated
from src.me.schemas import MeResponse, TokenSummary
from src.profiles import service as profiles_service
from src.tokens import service as tokens_service
from src.workflows import service as workflows_service


async def read_me(db: AsyncSession, user: AuthedUser) -> MeResponse:
    """The caller's account, profile, token summary, and supported hosts."""
    if user.token_id is None:  # pragma: no cover - only `require_token` reaches here
        raise unauthenticated()

    token = await tokens_service.get_token_by_id(db, user.token_id)

    if token is None:  # pragma: no cover - verified by `require_token` a moment ago
        raise unauthenticated()

    profile = await profiles_service.load_profile(db, user.id)
    supported_hosts = await workflows_service.get_supported_hosts(db)

    return MeResponse(
        user=UserOut(id=user.id, email=user.email),
        profile=profiles_service.to_profile_out(profile),
        profile_version=profile.version,
        token=TokenSummary(expires_at=token.expires_at, last4=token.last4),
        supported_hosts=supported_hosts,
    )
