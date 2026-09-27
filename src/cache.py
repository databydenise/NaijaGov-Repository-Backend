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
    max_entries: int | None = None,
) -> Any:  # noqa: ANN401  # the value type is the caller's, not this module's
    """
    The cached value for `key`, calling `load` when there is none or it has expired.

    Deliberately not locked: two requests arriving on a cold key both load, and the second
    overwrites the first with an identical value. A lock would cost every caller something
    to save a duplicate query that happens once per TTL.

    `max_entries` bounds how many keys may share this key's namespace — everything up to
    its last colon. Reference data does not need it, there being as many keys as there are
    workflows, but a cache keyed by something a caller chooses, such as a search query,
    grows as fast as requests arrive. Bounding per namespace rather than globally is what
    stops a flood of queries evicting the workflow registry and putting that load on the
    database instead.
    """
    now = time.monotonic()
    entry = _entries.get(key)

    if entry is not None and entry[0] > now:
        return entry[1]

    value = await load()

    if max_entries is not None and key not in _entries:
        _evict_to_fit(key.rpartition(":")[0], max_entries, now)

    _entries[key] = (now + ttl_seconds, value)

    return value


def _evict_to_fit(namespace: str, max_entries: int, now: float) -> None:
    """Make room in one namespace: expired entries first, then whichever expires soonest.

    Evicting the soonest-to-expire rather than the least recently used keeps this to one
    pass and no per-read bookkeeping. The cost of a wrong eviction is one recomputation.
    """
    prefix = f"{namespace}:"
    owned = [key for key in _entries if key.startswith(prefix)]

    if len(owned) < max_entries:
        return

    for key in owned:
        if _entries[key][0] <= now:
            del _entries[key]

    live = [key for key in owned if key in _entries]

    while len(live) >= max_entries:
        soonest = min(live, key=lambda key: _entries[key][0])
        del _entries[soonest]
        live.remove(soonest)


def invalidate(key: str | None = None) -> None:
    """Drop one key, or the whole cache. For tests, and after a seed in the same process."""
    if key is None:
        _entries.clear()

        return

    _entries.pop(key, None)
