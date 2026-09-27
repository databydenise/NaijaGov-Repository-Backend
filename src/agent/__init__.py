"""
The turn runner: the only code in this service that talks to a model.

Two entry points, and they are the whole public surface:

- `run_plan_turn` — the `/plan` turn. A rendered context from P2, this turn's page snapshot, the
  user's values, a retrieval callable and a model client go in; a guarded plan, its grounding
  verdict, the chunks retrieved, and a content-free telemetry record come out. Or a `TurnFailure`
  with a code and a sentence the side panel prints verbatim.
- `run_explain_turn` — the same machinery against the smaller explain schema, for P6.

Both are *returned* outcomes, never exceptions, and neither persists anything: P5 wraps the first
in `/plan` and decides what to store, P6 reuses the second for `/explain`.

Everything a turn may spend is a named constant in `constants.py`, because the difference between
a panel that explains itself and a panel that appears to have hung is a budget check. The model
client and the retrieval callable are injected, which is what lets every check in
`scripts/check_agent.py` run a whole turn — tool rounds, repairs, timeouts and all — with no API
key, no network and no database.
"""

from src.agent.constants import FAILURE_MESSAGES, FailureCode
from src.agent.explain import run_explain_turn
from src.agent.plan import run_plan_turn
from src.agent.schemas import (
    ExplainOutcome,
    ExplainTurn,
    PlanOutcome,
    PlanTurn,
    TurnFailure,
    TurnTelemetry,
)

__all__ = [
    "FAILURE_MESSAGES",
    "ExplainOutcome",
    "ExplainTurn",
    "FailureCode",
    "PlanOutcome",
    "PlanTurn",
    "TurnFailure",
    "TurnTelemetry",
    "run_explain_turn",
    "run_plan_turn",
]
