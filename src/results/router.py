"""
`POST /results`. Extension only, on `require_token`.

The handler authenticates, rate limits, delegates, and logs. Whether a run is recorded at all is
`service.py`'s decision; nothing here decides anything.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.exceptions.errors import rate_limited
from src.rate_limit import check_rate_limit
from src.results import service as results_service
from src.results.constants import RATE_LIMIT, RATE_WINDOW_SECONDS
from src.results.schemas import ResultsRequest, ResultsResponse
from src.tokens.dependencies import require_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["extension"])


@router.post("", response_model=ResultsResponse)
async def record_results(
    payload: ResultsRequest,
    user: Annotated[AuthedUser, Depends(require_token)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ResultsResponse:
    """Record what the extension actually managed to do, and say what the user does next."""
    limit = check_rate_limit(
        f"results:{user.token_id}",
        RATE_LIMIT,
        RATE_WINDOW_SECONDS,
    )

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    started = time.perf_counter()
    response = await results_service.record_results(db, user, payload)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    _log_request(user, payload, response, duration_ms)

    return response


def _log_request(
    user: AuthedUser,
    payload: ResultsRequest,
    response: ResultsResponse,
    duration_ms: float,
) -> None:
    """
    One line per request, metadata only.

    Everything here is an id from our own registry, a count, a code or a duration. There is no
    field label, no value, no URL — and on this endpoint there is nothing else to leak either,
    because the request has no field that could carry one.

    `elapsed_ms` is the extension's own measurement of the run and `duration_ms` is ours of this
    request. Both, because they answer different questions: how long the page took to fill, and
    how long the bookkeeping took.
    """
    run = response.run

    logger.info(
        "results user=%s session=%s plan=%s acknowledged=%s recorded=%d "
        "ok=%d changed=%d failed=%d rejected=%d cancelled=%d "
        "checkpoint=%s aborted=%s hint=%s step=%s elapsed_ms=%d duration_ms=%s",
        user.id,
        payload.session_id,
        payload.plan_id,
        response.acknowledged,
        response.recorded,
        run.ok,
        run.changed,
        run.failed,
        run.rejected,
        run.cancelled,
        payload.checkpoint.reason if payload.checkpoint else "-",
        payload.aborted or "-",
        response.next_hint.code,
        response.step.id if response.step else "-",
        payload.elapsed_ms,
        duration_ms,
    )
