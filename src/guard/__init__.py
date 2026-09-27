"""
The output guard: the last gate before anything reaches the fill preview.

`guard_plan` is the whole public surface. It takes a parsed `PlanResponse`, the snapshot the
request was made against, the user's real values, and the chunks retrieved this turn, and returns
a `GuardedPlan`: approved actions with their values substituted and their provenance named,
rejected actions with a code and a sentence for the user, verified citations only, the merged
`missing` list, and a grounding verdict.

Nothing here calls the model, touches the database, or runs the repair pass — P4 does that, using
`repair_requested`.
"""

from src.guard.constants import REJECTION_MESSAGES, UNVERIFIED_REPLY, RejectionCode
from src.guard.schemas import ApprovedAction, GuardedPlan, GuardReport, RejectedAction
from src.guard.service import guard_plan

__all__ = [
    "REJECTION_MESSAGES",
    "UNVERIFIED_REPLY",
    "ApprovedAction",
    "GuardReport",
    "GuardedPlan",
    "RejectedAction",
    "RejectionCode",
    "guard_plan",
]
