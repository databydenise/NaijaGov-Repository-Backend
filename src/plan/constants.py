"""
`/plan` limits, budgets, and the fixed strings a citizen reads.

Two catalogues here are contracts rather than tuning knobs:

- `PLAN_ERROR_MESSAGES` holds the sentences for the failures this endpoint owns rather than the
  runner: a session that is gone, and a page that changed under us.
- `SOURCE_LABELS` is how provenance is worded in the fill preview. The panel prints it beside
  every row, so it is written for a person: where the value came from, in five words.

How a *runner* failure becomes a status is not here: `/explain` fails the same six ways and must
answer them the same way, so that table lives in `src/turn_errors.py` and both endpoints read it.

The caps are shared with `/context` rather than restated. Both endpoints take the same snapshot
from the same content script, and two sets of limits on one payload shape is how they drift.
"""

from typing import Final

from src.agent.constants import TURN_BUDGET_SECONDS
from src.constants import ErrorCode

# --- Request ------------------------------------------------------------------------------
#
# What the user typed. Two thousand characters is a long question and far short of a pasted
# document; past it the request is refused rather than truncated, because a question we cut in
# half is a question we answer wrongly.
MAX_MESSAGE_CHARS: Final = 2000

# The panel's own correlation id, echoed back untouched. Capped because it is a string from a
# client and it reaches a response.
MAX_CLIENT_PLAN_ID_CHARS: Final = 64

# --- Rate limiting ------------------------------------------------------------------------
#
# Twenty plans a minute per token: well above human pace — a person reads a preview before
# asking again — and low enough that a stuck extension is stopped inside a minute. Per token
# rather than per user, so one broken install cannot spend another's budget.
RATE_LIMIT: Final = 20
RATE_WINDOW_SECONDS: Final = 60

# --- Budgets ------------------------------------------------------------------------------
#
# The whole request, measured from the moment the handler starts. This only ever *tightens* the
# runner's own twenty seconds: it is what stops a slow database turning a 20-second turn into a
# 25-second request that the panel has already given up on.
REQUEST_BUDGET_SECONDS: Final = TURN_BUDGET_SECONDS + 3.0

# Held back from the turn's deadline for the writes that follow it. A turn that used every last
# second and then found no time to save its own history would answer the user and forget the
# conversation, which is worse than answering half a second sooner.
PERSIST_RESERVE_SECONDS: Final = 2.0

# --- The idempotency window ---------------------------------------------------------------
#
# The same message, on the same page, in the same session, inside this window is the same
# question — a flaky connection retrying, not a person asking twice. Served from the stored
# plan, so a retry costs neither seconds nor money. Sixty seconds is long enough to cover a
# mobile timeout and short enough that "ask it again" still works as a way to get a fresh answer.
IDEMPOTENCY_WINDOW_SECONDS: Final = 60.0

# --- The pending plan --------------------------------------------------------------------
#
# How long an approved plan may be applied for. Ten minutes is enough to read a preview
# carefully and short enough that a plan cannot be applied to a page the user has since edited
# — the page hash catches that anyway, and this is the second bound.
PLAN_TTL_SECONDS: Final = 600.0

# A bound on both in-process stores, oldest evicted first. Nothing here is load-bearing for
# correctness — a retry that misses the cache costs a model call, and an approval that misses
# the store is refused — but an unbounded dict on a request path is a leak.
MAX_STORED_PLANS: Final = 500

# --- Provenance, as the preview words it -------------------------------------------------
#
# The guard hands us `profile.email` or `chat.lga`. The panel shows this instead, because
# "profile.email" is our word for it and not the user's. Every approved write has one of these
# beside it: a fill whose provenance cannot be named never reaches the preview at all.
SOURCE_LABELS: Final[dict[str, str]] = {
    "profile": "Your profile",
    "chat": "You told me just now",
}

# What a row with no traceable source is shown as. Unreachable from a guarded plan — the guard
# drops an unsourced write — and here so the mapping can never produce an empty provenance line.
UNKNOWN_SOURCE_LABEL: Final = "Not from your saved details"

# --- The failures this endpoint owns -----------------------------------------------------

PLAN_ERROR_MESSAGES: Final[dict[str, str]] = {
    ErrorCode.SESSION_NOT_FOUND: (
        "I've lost track of this page. Refresh it and I'll read it again."
    ),
    ErrorCode.PAGE_CHANGED: "The page changed while I was working. Let me read it again.",
}

# A step line for a page the registry does not know. `/context` answers such a page with
# `supported: false` and the panel still offers chat, so a plan turn has to be renderable
# without a workflow — with a step view that claims nothing about where the user is.
UNKNOWN_STEP_NAME: Final = "This page"
