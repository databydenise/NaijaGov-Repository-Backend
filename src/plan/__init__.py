"""
`POST /plan` — the endpoint behind PLANNING.

The panel sends the user's message and the page as it is now; this package authenticates, checks
the page has not changed, assembles the context, runs the turn, stores what the turn produced, and
returns a plan the user can approve. The judgement all happens elsewhere: P1 retrieves, P2 renders,
P3 guards, P4 runs the turn under its budgets.

Two things here are per-process and deliberately not in the database (`store.py`): the plans
awaiting approval, and the answers held for an identical retry. Both hold real values, both are
short-lived, and losing either on a restart costs a model call rather than a wrong answer.
"""

from src.plan.router import router
from src.plan.schemas import PlanRequest, PlanResponse

__all__ = ["PlanRequest", "PlanResponse", "router"]
