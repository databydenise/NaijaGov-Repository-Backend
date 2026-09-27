"""
A daily token ceiling per user.

This exists for one failure: a stuck extension retrying in a loop, turning a demo into an
invoice. It is not rationing a real user's help — the default is roughly fifty plan turns a day —
and the limit is checked *before* a turn rather than mid-way, so a user who is over it is told
plainly instead of having a turn cut in half.

Counted in memory, keyed by user and UTC date, and reset on restart. That is the same trade
`rate_limit.py` makes, and the spec accepts it for the quota explicitly. The consequence worth
knowing: a process restart clears everyone's spend, so this bounds a runaway loop within one
process lifetime, not a month's bill.

Nothing here holds a token, a prompt, or anything a user typed — only a user id and a count.
"""

import logging
from datetime import UTC, datetime

from src.agent.constants import QUOTA_WARN_FRACTION
from src.agent.exceptions import TurnAborted
from src.config import settings

logger = logging.getLogger(__name__)

_spent: dict[tuple[str, str], int] = {}


def _today() -> str:
    """The UTC date, as the key's second half. UTC so a day does not shift with a deployment."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def tokens_spent(user_id: str) -> int:
    """What this user has spent today."""
    return _spent.get((user_id, _today()), 0)


def check_quota(user_id: str) -> None:
    """
    Raise `TurnAborted("QUOTA_EXCEEDED")` when this user is out of tokens for the day.

    Checked before the first model call, so a refusal costs nothing. A turn already under way is
    never cut off by the quota — the tokens are spent either way, and half an answer is worse
    than one over the line.
    """
    quota = settings.daily_token_quota

    if quota <= 0:
        return

    spent = tokens_spent(user_id)

    if spent >= quota:
        logger.warning(
            "turn refused: daily token quota reached (user_id=%s spent=%d quota=%d)",
            user_id,
            spent,
            quota,
        )

        raise TurnAborted("QUOTA_EXCEEDED")


def record_usage(user_id: str, tokens: int) -> None:
    """
    Add a turn's tokens to today's count, and warn when a user is approaching the limit.

    Called for a failed turn as well as a successful one: a call that timed out after the
    provider had already read the prompt still cost what it cost.
    """
    if tokens <= 0:
        return

    key = (user_id, _today())
    spent = _spent.get(key, 0) + tokens
    _spent[key] = spent

    quota = settings.daily_token_quota

    if quota > 0 and spent >= quota * QUOTA_WARN_FRACTION:
        logger.warning(
            "user approaching daily token quota (user_id=%s spent=%d quota=%d)",
            user_id,
            spent,
            quota,
        )


def reset_quota() -> None:
    """Clear every counter. For checks and for a deliberate operational reset."""
    _spent.clear()
