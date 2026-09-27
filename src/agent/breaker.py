"""
A circuit breaker in front of the model client.

Without one, a provider outage costs every caller the full per-call timeout before they learn
anything: twelve seconds, twice more for the retries, and a panel that looks broken rather than
one that says the assistant is unreachable. After a few consecutive transport failures this
opens, and callers fail immediately for a short window instead.

Deliberately simple — a count and a timestamp, per process:

- Only *transport* failures count. A model that answered with unparseable content is not the
  provider being down, and opening the breaker on our own bad prompt would take the feature out
  for everyone.
- One success closes it. A half-open state with a trial quota is more machinery than a demo
  needs, and the window is short enough that the next caller through is the trial.
"""

import logging
import time

from src.agent.constants import BREAKER_FAILURE_THRESHOLD, BREAKER_OPEN_SECONDS

logger = logging.getLogger(__name__)

_consecutive_failures = 0
_opened_at: float | None = None


def is_open() -> bool:
    """Whether calls should fail fast right now. Closes itself once the window has passed."""
    global _opened_at, _consecutive_failures  # noqa: PLW0603  # one breaker per process

    if _opened_at is None:
        return False

    if time.monotonic() - _opened_at >= BREAKER_OPEN_SECONDS:
        logger.info("model circuit breaker window elapsed; trying again")
        _opened_at = None
        _consecutive_failures = 0

        return False

    return True


def record_failure() -> None:
    """Count one transport failure, and open the breaker at the threshold."""
    global _opened_at, _consecutive_failures  # noqa: PLW0603

    _consecutive_failures += 1

    if _consecutive_failures >= BREAKER_FAILURE_THRESHOLD and _opened_at is None:
        _opened_at = time.monotonic()
        logger.error(
            "model circuit breaker opened after %d consecutive failures; "
            "failing fast for %.0fs",
            _consecutive_failures,
            BREAKER_OPEN_SECONDS,
        )


def record_success() -> None:
    """The provider answered. Forget the failures."""
    global _opened_at, _consecutive_failures  # noqa: PLW0603

    _consecutive_failures = 0
    _opened_at = None


def reset_breaker() -> None:
    """Close the breaker and clear its count. For checks and for an operational reset."""
    record_success()
