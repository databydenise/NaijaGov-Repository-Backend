"""
`POST /explain`. Extension only, on `require_token`.

The handler authenticates, rate limits, delegates, and logs. Every decision — sensitive or not,
cached or not, retrieved or not, model or no model — is in `service.py`; nothing here decides
anything.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.exceptions.errors import rate_limited
from src.explain import metrics
from src.explain import service as explain_service
from src.explain.constants import RATE_LIMIT, RATE_WINDOW_SECONDS
from src.explain.schemas import ExplainRequest, ExplainResponse
from src.rate_limit import check_rate_limit
from src.tokens.dependencies import require_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["extension"])


@router.post("", response_model=ExplainResponse)
async def explain_field(
    payload: ExplainRequest,
    user: Annotated[AuthedUser, Depends(require_token)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ExplainResponse:
    """Explain one field on this page, in plain English, with a source where there is one."""
    limit = check_rate_limit(
        f"explain:{user.token_id}",
        RATE_LIMIT,
        RATE_WINDOW_SECONDS,
    )

    if not limit.allowed:
        raise rate_limited(limit.retry_after_seconds)

    started = time.perf_counter()
    response = await explain_service.explain_field(db, user, payload)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    _log_request(user, payload, response, duration_ms)

    return response


def _log_request(
    user: AuthedUser,
    payload: ExplainRequest,
    response: ExplainResponse,
    duration_ms: float,
) -> None:
    """
    One line per request, metadata only.

    `field_id` is here because it is the content script's own handle for a control — the same thing
    `action_log` stores — and it is what makes "which field was slow" answerable. The field's
    **label** is not, and neither is the user's question: a label on a government form can name a
    medical condition or a benefit, and a question about an application is the most sensitive
    sentence a citizen will type into this product. Both appear as a length or not at all.

    `cache_hit_rate` is definition-of-done item 5. It is the running figure for this process, so a
    demo page with six fields should show it climbing towards 1.0 on the second pass — which is how
    you know the second run will feel instant before anyone is watching.
    """
    stats = metrics.cache_stats()
    rate = stats.hit_rate

    logger.info(
        "explain user=%s session=%s field=%s question_chars=%d nearby_chars=%d "
        "grounded=%s cached=%s sources=%d cache_hits=%d/%d cache_hit_rate=%s duration_ms=%s",
        user.id,
        payload.session_id or "-",
        payload.field.field_id,
        len(payload.question or ""),
        len(payload.nearby_text),
        response.grounded,
        response.cached,
        len(response.sources),
        stats.hits,
        stats.lookups,
        f"{rate:.2f}" if rate is not None else "-",
        duration_ms,
    )
