"""
A failed model turn, as an HTTP answer. Shared by every endpoint that runs one.

The runner deliberately does not know about HTTP — `agent/constants.py` says so, and it is right
to: a turn that ran out of budget is the same event whether it was asked for by `/plan`, by
`/explain`, or by a script. Something still has to decide what a caller is told, and this is that
something, in one place rather than one copy per endpoint.

One place, because the alternative was two: `/plan` and `/explain` fail in exactly the same ways
(the same provider, the same budgets, the same lock, the same quota), and two tables that must
agree are two tables that eventually will not. A caller branching on `MODEL_TIMEOUT` should not
have to know which endpoint it asked.

The sentence the panel prints is never written here. It comes from `FAILURE_MESSAGES` in
`agent/constants.py`, where the code that diagnosed the failure lives; this layer only knows it is
speaking HTTP.
"""

from typing import Final

from fastapi import status

from src.agent.schemas import TurnFailure
from src.constants import ErrorCode
from src.exceptions.errors import api_error

# Every runner failure code, and the status a caller is answered with.
#
# Two are mapped to a status that does not repeat their own name, and both are deliberate:
#
# - `BUDGET_EXCEEDED` → 504. From outside, a turn that ran out of time and a provider that never
#   answered are the same event: we waited, and there is no answer. The codes stay distinct in the
#   log line, where the difference is actionable.
# - `SCHEMA_FAILED` → 502 `PLAN_FAILED`. The provider answered; what it said could not be made
#   into an answer even after a repair. That is an upstream fault, not a timeout and not the
#   caller's request being wrong.
FAILURE_STATUS: Final[dict[str, int]] = {
    "MODEL_UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
    "MODEL_TIMEOUT": status.HTTP_504_GATEWAY_TIMEOUT,
    "BUDGET_EXCEEDED": status.HTTP_504_GATEWAY_TIMEOUT,
    "SCHEMA_FAILED": status.HTTP_502_BAD_GATEWAY,
    "SESSION_BUSY": status.HTTP_409_CONFLICT,
    "QUOTA_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS,
}

# The code returned for each runner failure, where it differs from the runner's own name.
FAILURE_CODES: Final[dict[str, str]] = {
    "BUDGET_EXCEEDED": ErrorCode.MODEL_TIMEOUT,
    "SCHEMA_FAILED": ErrorCode.PLAN_FAILED,
}


def turn_failed(failure: TurnFailure) -> Exception:
    """
    The runner's failure, as an HTTP answer carrying the runner's own sentence.

    An unmapped code — one added to the runner without a status here — is answered as a 503 with
    the runner's sentence, rather than raising a `KeyError` that becomes a 500 with no sentence at
    all. The user is told something true either way; the log line carries the code.
    """
    return api_error(
        FAILURE_STATUS.get(failure.code, status.HTTP_503_SERVICE_UNAVAILABLE),
        FAILURE_CODES.get(failure.code, failure.code),
        failure.message,
    )
