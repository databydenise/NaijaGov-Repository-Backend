"""
In-memory TTL cache.

For reference data that changes only when the seed runs: the workflow registry and the
counts derived from it. Per process and not shared between workers, which is right for
this — a worker holding a five-minute-old copy of a table nobody wrote to is not a
problem worth a Redis for.

Never used for anything user-scoped. A cache keyed by user is one bug away from serving
one person's data to another, and nothing here is worth that risk.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

_entries: dict[str, tuple[float, Any]] = {}


async def cached(
    key: str,
    ttl_seconds: float,
    load: Callable[[], Awaitable[Any]],
) -> Any:  # noqa: ANN401  # the value type is the caller's, not this module's
    """
    The cached value for `key`, calling `load` when there is none or it has expired.

    Deliberately not locked: two requests arriving on a cold key both load, and the second
    overwrites the first with an identical value. A lock would cost every caller something
    to save a duplicate query that happens once per TTL.
    """
    now = time.monotonic()
    entry = _entries.get(key)

    if entry is not None and entry[0] > now:
        return entry[1]

    value = await load()
    _entries[key] = (now + ttl_seconds, value)

    return value


def invalidate(key: str | None = None) -> None:
    """Drop one key, or the whole cache. For tests, and after a seed in the same process."""
    if key is None:
        _entries.clear()

        return

    _entries.pop(key, None)
