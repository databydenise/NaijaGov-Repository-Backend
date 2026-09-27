"""
What both entry points share: the turn's state, its deadline, and its failure path.

Small on purpose. `plan.py` and `explain.py` hold the two turns; this holds the three things they
would otherwise each own a copy of, and the copies would drift — most damagingly the failure path,
which is the one place that decides what a citizen is told when a turn cannot finish.
"""

import logging
import time

from src.agent.constants import TURN_BUDGET_SECONDS
from src.agent.exceptions import TurnAborted
from src.agent.schemas import TurnFailure
from src.agent.telemetry import TurnState, build_telemetry, log_turn
from src.ai.client import ModelClient
from src.config import settings

logger = logging.getLogger(__name__)

PLAN_SCHEMA_NAME = "plan_response"
EXPLAIN_SCHEMA_NAME = "explain_response"


def new_state(kind: str, client: ModelClient, session_id: str, user_id: str) -> TurnState:
    """The turn's accumulator. The model name comes from the client, so a fake says so."""
    return TurnState(
        kind=kind,
        model=getattr(client, "model", settings.model_name),
        session_id=session_id,
        user_id=user_id,
    )


def turn_deadline_from(given: float | None) -> float:
    """The turn's deadline on the monotonic clock. A caller may impose a tighter one."""
    return given if given is not None else time.monotonic() + TURN_BUDGET_SECONDS


def as_failure(abort: TurnAborted, state: TurnState) -> TurnFailure:
    """A failed turn, measured and logged like any other."""
    record = build_telemetry(state, outcome=abort.code)
    log_turn(record)

    return TurnFailure(code=abort.code, message=abort.message, telemetry=record)
