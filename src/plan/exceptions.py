"""
The failures `/plan` returns, as error-contract factories.

Every one of these carries a sentence written for a citizen, because the side panel prints
`message` verbatim. None carries a stack trace, a model response, a field label, or anything the
user typed.

Only the two failures this endpoint owns are here. A failed *turn* is not one of them: every
endpoint that runs a turn fails the same six ways, so `turn_failed` lives in `src/turn_errors.py`
and is re-exported here, because `/plan`'s own routes should not have to know that.
"""

from fastapi import status

from src.constants import ErrorCode
from src.exceptions.errors import api_error
from src.plan.constants import PLAN_ERROR_MESSAGES
from src.turn_errors import turn_failed

__all__ = ["page_changed", "session_not_found", "turn_failed"]


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
