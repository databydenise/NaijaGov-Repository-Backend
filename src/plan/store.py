"""
Two in-process stores: plans awaiting approval, and plans already answered.

Both hold real profile values, so both are deliberately *not* in `src/cache.py` — that cache is
shared across namespaces and documented as never for user-scoped data. They are held here
instead, in this process only, keyed by session and user, and swept on every write.

In memory and per worker, like `rate_limit.py`, `agent/locks.py` and `agent/quota.py`. The
consequences are worth knowing rather than discovering:

- **A restart drops both.** A pending plan is then refused on approval and the user asks again;
  a cached answer is recomputed at the cost of one model call. Neither is a wrong answer, which
  is the property that makes this trade acceptable — and `fastapi dev` restarts on every save,
  so during development it happens constantly.
- **Two workers do not share them.** An approval that lands on the other worker is refused, and
  a retry that lands there costs a model call. Moving both to Postgres or Redis is the first
  thing a deployed service needs.

The idempotency key holds a **hash** of the message, never the message. The question a citizen
asked is not something to keep in a process's memory as a dictionary key where a heap dump or a
debugger would read it back — and a hash compares just as well.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass

from src.plan.constants import (
    IDEMPOTENCY_WINDOW_SECONDS,
    MAX_STORED_PLANS,
    PLAN_TTL_SECONDS,
)
from src.plan.schemas import PlannedActionOut, PlanResponse


@dataclass(frozen=True)
class PendingPlan:
    """
    The approved actions of one plan, waiting for the user to accept them.

    Scoped to a user as well as a session: a plan id is unguessable, but an authorisation check
    that relies on a value being unguessable is not an authorisation check.
    """

    plan_id: str
    session_id: uuid.UUID
    user_id: uuid.UUID
    actions: tuple[PlannedActionOut, ...]
    expires_at: float


@dataclass(frozen=True)
class _CachedPlan:
    """One answered plan, for an identical retry. `stored_at` is on the monotonic clock."""

    response: PlanResponse
    stored_at: float


# plan_id → the actions it approved.
_pending: dict[str, PendingPlan] = {}

# (session_id, page_hash, message hash) → the answer that was given.
_answered: dict[tuple[str, str, str], _CachedPlan] = {}


def new_plan_id() -> str:
    """A fresh plan id. A uuid4 hex: unguessable, and it carries nothing about the plan."""
    return uuid.uuid4().hex


def idempotency_key(session_id: uuid.UUID, page_hash: str, message: str) -> tuple[str, str, str]:
    """
    The key three identical requests share.

    The message is hashed, not held: see the module docstring. Stripped and case-folded first, so
    a retry that differs only in trailing whitespace is recognised as the same question.
    """
    digest = hashlib.sha256(message.strip().casefold().encode("utf-8")).hexdigest()

    return (str(session_id), page_hash, digest)


def _sweep(now: float) -> None:
    """Drop what has expired from both stores, and cap what is left."""
    for plan_id in [key for key, plan in _pending.items() if plan.expires_at <= now]:
        del _pending[plan_id]

    for key in [
        key
        for key, cached in _answered.items()
        if now - cached.stored_at >= IDEMPOTENCY_WINDOW_SECONDS
    ]:
        del _answered[key]

    # Oldest first, so what survives a flood is what is most likely still being looked at.
    while len(_pending) > MAX_STORED_PLANS:
        oldest = min(_pending, key=lambda key: _pending[key].expires_at)
        del _pending[oldest]

    while len(_answered) > MAX_STORED_PLANS:
        oldest = min(_answered, key=lambda key: _answered[key].stored_at)
        del _answered[oldest]


def remember_plan(
    plan_id: str,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    actions: tuple[PlannedActionOut, ...],
) -> None:
    """Hold this plan's approved actions for `PLAN_TTL_SECONDS`."""
    now = time.monotonic()
    _sweep(now)

    _pending[plan_id] = PendingPlan(
        plan_id=plan_id,
        session_id=session_id,
        user_id=user_id,
        actions=actions,
        expires_at=now + PLAN_TTL_SECONDS,
    )


def get_pending_plan(plan_id: str, user_id: uuid.UUID) -> PendingPlan | None:
    """
    The pending plan held for this user, or None if it is unknown, expired, or someone else's.

    One answer for all three, for the reason `unauthenticated()` gives one answer for eight
    causes: the differences are only useful to someone probing.
    """
    plan = _pending.get(plan_id)

    if plan is None:
        return None

    if plan.expires_at <= time.monotonic() or plan.user_id != user_id:
        return None

    return plan


def remember_answer(key: tuple[str, str, str], response: PlanResponse) -> None:
    """Keep this answer for an identical retry inside the idempotency window."""
    now = time.monotonic()
    _sweep(now)

    _answered[key] = _CachedPlan(response=response, stored_at=now)


def get_answer(key: tuple[str, str, str]) -> PlanResponse | None:
    """The answer already given to this exact request, if it is still inside the window."""
    cached = _answered.get(key)

    if cached is None:
        return None

    if time.monotonic() - cached.stored_at >= IDEMPOTENCY_WINDOW_SECONDS:
        del _answered[key]

        return None

    return cached.response


def reset_plans() -> None:
    """Clear both stores. For checks and for a deliberate operational reset."""
    _pending.clear()
    _answered.clear()


@dataclass(frozen=True)
class StoreSizes:
    """What both stores hold, for a check or a health line. Counts only."""

    pending: int
    answered: int


def sizes() -> StoreSizes:
    """How many plans are held. Never what is in them."""
    return StoreSizes(pending=len(_pending), answered=len(_answered))
