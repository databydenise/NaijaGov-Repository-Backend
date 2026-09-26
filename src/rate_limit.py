"""
In-memory rate limiting.

A fixed window per key, held in this process. That is the right size for a demo: it needs no
Redis, and its only weakness — counters reset on restart and are not shared between workers
— is acceptable when the thing being limited is a login form.

Not a substitute for the per-user and per-token limits the deployed service will need on
`/plan`. A stuck extension retrying in a loop must not become a bill, and that limit belongs
in a middleware with a shared store.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass

_hits: dict[str, deque[float]] = defaultdict(deque)


@dataclass(frozen=True)
class RateLimitResult:
    """Whether the call is allowed, and how long to wait if not."""

    allowed: bool
    retry_after_seconds: int


def check_rate_limit(key: str, limit: int, window_seconds: int) -> RateLimitResult:
    """
    Record an attempt against `key` and say whether it is allowed.

    The key is caller-chosen and must never be a secret: it ends up in this process's
    memory, so it holds an email or a user id, never a password or a token.
    """
    now = time.monotonic()
    window_start = now - window_seconds
    attempts = _hits[key]

    while attempts and attempts[0] < window_start:
        attempts.popleft()

    if len(attempts) >= limit:
        retry_after = int(attempts[0] + window_seconds - now) + 1

        return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

    attempts.append(now)

    return RateLimitResult(allowed=True, retry_after_seconds=0)


def reset_rate_limits() -> None:
    """Clear every counter. For tests and for a deliberate operational reset."""
    _hits.clear()
