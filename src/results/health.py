"""
What `/health` says about the run counters.

Separate from `service.py` for the reason `documents/health.py` is: the write path imports nothing
it does not need, and the reporting can answer even when a write could not.

This is the read side the spec asks for — "not a dashboard; a table". It is on `/health` rather
than behind an admin route because no admin or staff auth exists anywhere in this project, and
inventing one to show a count of fills would be a bigger decision than the counts are worth.
What it publishes is aggregate and content-free: how many actions reached each status, per
workflow, per step, per day.
"""

import logging

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.cache import cached
from src.results.constants import (
    COUNTERS_CACHE_KEY,
    COUNTERS_CACHE_TTL_SECONDS,
    MAX_HEALTH_COUNTERS,
)
from src.results.models import ResultsCounter

logger = logging.getLogger(__name__)


async def get_counters(db: AsyncSession) -> list[dict[str, object]]:
    """
    The most recent counters, newest day first.

    Bounded by `MAX_HEALTH_COUNTERS`: a liveness check that grows with a year of history stops
    being a liveness check. Ordered by day so what comes back is the current shape of things
    rather than an arbitrary window of it.
    """
    statement = (
        select(
            ResultsCounter.day,
            ResultsCounter.workflow_id,
            ResultsCounter.step_id,
            ResultsCounter.metric,
            ResultsCounter.count,
        )
        .order_by(desc(ResultsCounter.day), ResultsCounter.workflow_id, ResultsCounter.metric)
        .limit(MAX_HEALTH_COUNTERS)
    )

    rows = (await db.execute(statement)).all()

    return [
        {
            "day": row.day.isoformat(),
            "workflow": row.workflow_id,
            "step": row.step_id,
            "metric": row.metric,
            "count": row.count,
        }
        for row in rows
    ]


async def get_cached_counters(db: AsyncSession) -> list[dict[str, object]]:
    """The counters, cached for five minutes. `/health` is polled; these change per run."""

    async def load() -> list[dict[str, object]]:
        return await get_counters(db)

    return await cached(COUNTERS_CACHE_KEY, COUNTERS_CACHE_TTL_SECONDS, load)


async def results_health(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[dict[str, object]] | None:
    """
    The `results` block of `/health`.

    Never raises, and degrades to `null` rather than an empty list: "the database did not answer"
    and "nothing has been reported yet" are different facts, and an empty list is the second one.
    """
    try:
        async with session_factory() as db:
            return await get_cached_counters(db)
    except Exception as exc:  # noqa: BLE001  # liveness must survive a database failure
        logger.warning("Could not read the results counters (%s)", type(exc).__name__)

        return None
