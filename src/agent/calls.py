"""
One model call, inside the turn's budget.

The client makes one HTTP attempt and knows nothing about time; this is where the policy lives:
how long this call may take given what is left of the turn, whether a failure is worth repeating,
how long to wait before repeating it, and when to stop asking a provider that is plainly down.

The order of the guards matters and is the order they appear in:

1. **Budget first.** Under two seconds left, nothing starts. A twelve-second call begun with one
   second of budget produces a timeout the user waits for and learns nothing from.
2. **Breaker second.** If the provider has failed three times in a row, this caller is not made
   to prove it again at twelve seconds a go.
3. **Then attempts.** Transient failures only, jittered, and never past the budget.

Every abort raises `TurnAborted` with the code the panel will show. Nothing here logs a message,
a prompt, or a reply — only counts, types and durations.
"""

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from typing import Any

from src.agent.breaker import is_open, record_failure, record_success
from src.agent.constants import (
    MAX_TRANSIENT_RETRIES,
    MIN_PHASE_SECONDS,
    MODEL_CALL_TIMEOUT_SECONDS,
    RETRY_BASE_SECONDS,
    RETRY_MAX_SECONDS,
)
from src.agent.exceptions import TurnAborted
from src.agent.telemetry import TurnState, phase
from src.ai.client import (
    Message,
    ModelClient,
    ModelPermanentError,
    ModelReply,
    ModelTimeoutError,
    ModelTransportError,
)

logger = logging.getLogger(__name__)


def remaining_seconds(deadline: float) -> float:
    """How much of the turn is left, on the monotonic clock the deadline was made with."""
    return deadline - time.monotonic()


def has_budget(deadline: float, need: float = MIN_PHASE_SECONDS) -> bool:
    """Whether a phase needing `need` seconds may start."""
    return remaining_seconds(deadline) >= need


def _backoff_seconds(attempt: int) -> float:
    """
    Jittered exponential backoff for attempt `attempt` (0-based).

    Full jitter across the interval, not a fixed delay plus noise: the failure mode this protects
    against is every extension that failed in the same second coming back in the same second.
    """
    ceiling = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**attempt))

    return random.uniform(RETRY_BASE_SECONDS / 2, ceiling)  # noqa: S311  # backoff, not crypto


async def call_model(
    *,
    client: ModelClient,
    state: TurnState,
    messages: Sequence[Message],
    deadline: float,
    tools: Sequence[dict[str, Any]] = (),
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "response",
) -> ModelReply:
    """
    Make one model call, retrying only what is worth retrying.

    Raises `TurnAborted` with `BUDGET_EXCEEDED` if there is not enough time to start,
    `MODEL_TIMEOUT` if the provider never answered in time, or `MODEL_UNAVAILABLE` if it could
    not be reached at all — including when the breaker is already open.
    """
    if not has_budget(deadline):
        logger.info("model call skipped: %.1fs budget left", remaining_seconds(deadline))

        raise TurnAborted("BUDGET_EXCEEDED")

    if is_open():
        raise TurnAborted("MODEL_UNAVAILABLE")

    last_code = "MODEL_UNAVAILABLE"

    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        timeout = min(MODEL_CALL_TIMEOUT_SECONDS, remaining_seconds(deadline))

        try:
            with phase(state, "model"):
                reply = await client.complete(
                    messages=messages,
                    tools=tools,
                    response_schema=response_schema,
                    schema_name=schema_name,
                    timeout_seconds=timeout,
                )
        except ModelTimeoutError:
            record_failure()
            last_code = "MODEL_TIMEOUT"
        except ModelTransportError:
            record_failure()
            last_code = "MODEL_UNAVAILABLE"
        except ModelPermanentError:
            # No key, a rejected schema, a bad request: asking again cannot help, and the user's
            # budget is better spent telling them than proving it.
            record_failure()

            raise TurnAborted("MODEL_UNAVAILABLE") from None
        else:
            record_success()
            state.record_reply(reply)

            return reply

        if attempt == MAX_TRANSIENT_RETRIES:
            break

        delay = _backoff_seconds(attempt)

        if remaining_seconds(deadline) - delay < MIN_PHASE_SECONDS:
            # There is no time for another attempt, so report the failure that caused this
            # rather than the budget: the provider is what went wrong.
            logger.info("no budget for a retry after %s", last_code)

            break

        state.retries += 1
        await asyncio.sleep(delay)

    raise TurnAborted(last_code)  # type: ignore[arg-type]  # both values are FailureCodes
