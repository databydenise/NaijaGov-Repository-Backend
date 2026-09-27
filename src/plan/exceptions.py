"""
The failures `/plan` returns, as error-contract factories.

Every one of these carries a sentence written for a citizen, because the side panel prints
`message` verbatim. None carries a stack trace, a model response, a field label, or anything the
user typed.

The runner's failures are translated rather than re-worded: `agent/constants.py` already holds
the sentence for each of its codes, and writing a second set here would mean two places to change
and one of them forgotten.
"""

from fastapi import status

from src.agent.schemas import TurnFailure
from src.constants import ErrorCode
from src.exceptions.errors import api_error
from src.plan.constants import FAILURE_CODES, FAILURE_STATUS, PLAN_ERROR_MESSAGES


def session_not_found() -> Exception:
    """
    404 for a session that is unknown, expired, or another user's.

    One answer for all three. A 403 for someone else's session would confirm that the id exists,
    which is the only thing an attacker guessing ids wants to learn.
    """
    return api_error(
        status.HTTP_404_NOT_FOUND,
        ErrorCode.SESSION_NOT_FOUND,
        PLAN_ERROR_MESSAGES[ErrorCode.SESSION_NOT_FOUND],
    )


def page_changed() -> Exception:
    """
    409 when the snapshot no longer hashes to what the session recorded.

    A refusal rather than a best effort, and the most important error here: filling a form from a
    snapshot of a different page is the worst outcome this endpoint can produce.
    """
    return api_error(
        status.HTTP_409_CONFLICT,
        ErrorCode.PAGE_CHANGED,
        PLAN_ERROR_MESSAGES[ErrorCode.PAGE_CHANGED],
    )


def turn_failed(failure: TurnFailure) -> Exception:
    """
    The runner's failure, as this endpoint's HTTP answer.

    An unmapped code — one added to the runner without a status here — is answered as a 503 with
    the runner's own sentence, rather than raising a `KeyError` that becomes a 500 with no
    sentence at all. The user is told something true either way; the log line carries the code.
    """
    return api_error(
        FAILURE_STATUS.get(failure.code, status.HTTP_503_SERVICE_UNAVAILABLE),
        FAILURE_CODES.get(failure.code, failure.code),
        failure.message,
    )
