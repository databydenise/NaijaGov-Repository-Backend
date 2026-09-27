"""
One in-flight turn per session.

Two plans racing against one page produce two fill previews for the same fields, and the user
applies whichever arrived second. So a second turn for a session that is already running is
refused — `SESSION_BUSY`, with a sentence asking them to wait — rather than queued behind a turn
that may itself take twenty seconds.

In memory and per process, like `rate_limit.py` and `cache.py`. That is honest for a single
worker and wrong for several: two workers would each allow one turn. Moving this to a row lock or
Redis is on the list for a deployed service; for a demo the extension only has one panel open per
tab anyway.

No `asyncio.Lock`: a lock's job is to make the second caller *wait*, which is exactly what must
not happen here. A set membership test is the whole mechanism, and it is atomic because nothing
between the read and the write awaits.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from src.agent.exceptions import TurnAborted

logger = logging.getLogger(__name__)

_in_flight: set[str] = set()


@asynccontextmanager
async def turn_lock(session_id: str) -> AsyncIterator[None]:
    """
    Hold the turn slot for one session, or raise `TurnAborted("SESSION_BUSY")`.

    Released in a `finally`, so a failure anywhere in the turn — including an unexpected one that
    propagates past the runner — cannot leave a session unable to ask anything again.
    """
    if session_id in _in_flight:
        logger.info("turn refused: session already running (session_id=%s)", session_id)

        raise TurnAborted("SESSION_BUSY")

    _in_flight.add(session_id)

    try:
        yield
    finally:
        _in_flight.discard(session_id)


def is_running(session_id: str) -> bool:
    """Whether a turn holds this session's slot. For checks and for `/health`-style reporting."""
    return session_id in _in_flight


def reset_locks() -> None:
    """Release every slot. For checks, and for a deliberate operational reset."""
    _in_flight.clear()
