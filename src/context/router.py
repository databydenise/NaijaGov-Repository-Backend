"""
`POST /context`. Extension only, on `require_token`.

The handler authenticates, rate limits, delegates, and logs. The judgement is in
`knowledge/match.py` and the orchestration in `service.py`; nothing here decides anything
about a page.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.context import service as context_service
from src.context.constants import RATE_LIMIT, RATE_WINDOW_SECONDS
from src.context.schemas import ContextRequest, ContextResponse
from src.context.utils import host_of
from src.database.session import get_session
from src.exceptions.errors import rate_limited
from src.rate_limit import check_rate_limit
from src.tokens.dependencies import require_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["extension"])


@router.post("", response_model=ContextResponse)
async def read_context(
    payload: ContextRequest,
    user: Annotated[AuthedUser, Depends(require_token)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ContextResponse:
    """Identify the page in the snapshot and open or refresh this tab's session."""
    limit = check_rate_limit(
        f"context:{user.token_id}",
        RATE_LIMIT,
        RATE_WINDOW_SECONDS,
    )

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    started = time.perf_counter()
    response = await context_service.read_context(db, user.id, payload)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    _log_request(user, payload, response, duration_ms)

    return response


def _log_request(
    user: AuthedUser,
    payload: ContextRequest,
    response: ContextResponse,
    duration_ms: float,
) -> None:
    """
    One line per request, metadata only.

    Every value here is a count, an id from our own registry, or a host. No label, no
    heading, no title, no option, no full URL, and nothing from `sensitive_flags` beyond
    how many there were — a government portal puts an application number in a query
    string, and a field label can name a medical condition.
    """
    logger.info(
        "context user=%s tab=%s host=%s workflow=%s step=%s confidence=%s "
        "fields=%d checkpoints=%d cached=%s rules=%d duration_ms=%s",
        user.id,
        payload.tab_id,
        host_of(payload.url),
        response.workflow.id if response.workflow else "-",
        response.step.id if response.step else "-",
        response.confidence or "-",
        len(payload.fields),
        response.checkpoint.count if response.checkpoint else 0,
        response.cached,
        response.rules_loaded,
        duration_ms,
    )
