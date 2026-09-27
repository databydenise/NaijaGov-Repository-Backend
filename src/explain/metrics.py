"""
The cache hit rate, for the log line.

Definition of done item 5 is "the cache hit rate is visible in the logs", and this is the whole of
it: two counters and a ratio, per process. On a demo page with six fields the rate should be near
total after the first pass, and seeing that in the log is how you know the second run on stage will
feel instant *before* you are standing in front of anyone.

Counted here rather than on the row, deliberately. A hit count on the cached row would mean an
UPDATE on every read — turning the cheapest path in this service into a write, and sliding the
row's own timestamps forward while it did so.

Per process and reset on restart, like `rate_limit.py` and `agent/quota.py`. A number that only
covers this worker since its last reload is exactly what the question "is the cache working?"
wants; an all-time figure across workers would answer a different question, and need a store.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_hits = 0
_misses = 0


@dataclass(frozen=True)
class CacheStats:
    """Lookups this process has made, and how many were answered from the table."""

    hits: int
    misses: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Hits per lookup, or None before any lookup has been made."""
        if not self.lookups:
            return None

        return self.hits / self.lookups


def record_lookup(*, hit: bool) -> None:
    """
    Record one cache lookup's outcome.

    Only a real lookup counts. A sensitive field and a custom question never consult the cache, so
    counting them would make the rate a measure of how often the cache was *asked*, which is not
    the number anyone wants when they ask whether it is working.
    """
    global _hits, _misses  # noqa: PLW0603  # two counters, one process, no instance to hang them on

    if hit:
        _hits += 1
    else:
        _misses += 1


def cache_stats() -> CacheStats:
    """The counters as they stand."""
    return CacheStats(hits=_hits, misses=_misses)


def reset_cache_stats() -> None:
    """Clear both counters. For checks, and for a deliberate operational reset."""
    global _hits, _misses  # noqa: PLW0603  # see `record_lookup`

    _hits = 0
    _misses = 0
