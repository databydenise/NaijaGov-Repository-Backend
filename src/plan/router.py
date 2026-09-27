"""
`POST /plan`. Extension only, on `require_token`.

The handler authenticates, rate limits, delegates, and logs. Every decision about the page, the
model and the plan is in `service.py` and in the packages it calls; nothing here decides anything.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.context.utils import host_of
from src.database.session import get_session
from src.exceptions.errors import rate_limited
from src.plan import service as plan_service
from src.plan.constants import RATE_LIMIT, RATE_WINDOW_SECONDS
from src.plan.schemas import PlanRequest, PlanResponse
from src.rate_limit import check_rate_limit
from src.tokens.dependencies import require_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plan", tags=["extension"])


@router.post("", response_model=PlanResponse)
async def create_plan(
    payload: PlanRequest,
    user: Annotated[AuthedUser, Depends(require_token)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> PlanResponse:
    """Plan a turn for this page and message, and return a preview the user can approve."""
    limit = check_rate_limit(
        f"plan:{user.token_id}",
        RATE_LIMIT,
        RATE_WINDOW_SECONDS,
    )

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    started = time.perf_counter()
    response = await plan_service.create_plan(db, user, payload)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    _log_request(user, payload, response, duration_ms)

    return response


def _log_request(
    user: AuthedUser,
    payload: PlanRequest,
    response: PlanResponse,
    duration_ms: float,
) -> None:
    """
    One line per request, metadata only.

    Everything here is an id from our own registry, a count, a verdict or a duration. The
    message appears as a **length**; the reply, the values, the labels and the retrieved text
    appear not at all. A log line that carries a profile value is a privacy incident whatever the
    log level says, and on this endpoint the values are real.

    The model's own numbers — calls, tokens, phases, repairs, cost — are not repeated here.
    `agent/telemetry.py` emits them as its own line carrying the same session id, so the two
    correlate; copying them would double the volume and give the second copy a chance to be wrong.
    """
    logger.info(
        "plan user=%s session=%s host=%s step=%s cached=%s message_chars=%d fields=%d "
        "approved=%d rejected=%d missing=%d citations=%d grounding=%s plan=%s duration_ms=%s",
        user.id,
        payload.session_id,
        host_of(payload.url),
        response.step.id if response.step else "-",
        response.cached,
        len(payload.message),
        len(payload.fields),
        len(response.actions),
        len(response.rejected),
        len(response.missing),
        len(response.citations),
        response.grounding,
        response.plan_id,
        duration_ms,
    )
