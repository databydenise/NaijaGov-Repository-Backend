"""
The runner's budgets, caps, failure catalogue, and cost table.

Everything a turn is allowed to spend is a named constant here, with the reasoning beside it.
That is not tidiness: the numbers below are the difference between a panel that says "taking too
long, try again" and a panel that appears to have hung, and someone tuning them during a demo
needs to see what each one protects.

Two catalogues here are contracts rather than tuning knobs:

- `FAILURE_MESSAGES` pairs every failure code with the sentence the side panel prints verbatim.
  Plain English, and a next step where there is one. Adding a code is something the extension
  has to render.
- `MODEL_RATES` is what turns token counts into a cost estimate. Its figures are **not set**;
  see the comment there. A wrong number in a bill estimate is worse than no number.
"""

from dataclasses import dataclass
from typing import Final, Literal

from src.documents.constants import (
    EMBEDDING_TIMEOUT_SECONDS,
    SEARCH_STATEMENT_TIMEOUT_MS,
)

# --- The whole turn -----------------------------------------------------------------
#
# Twenty seconds is the outer edge of what someone watching a side panel will wait. Every phase
# checks what is left before it starts, so running out is a reported outcome and never a hang.
TURN_BUDGET_SECONDS: Final = 20.0

# One model call. Three calls at twelve seconds cannot all fit in the turn budget, which is the
# point: the budget check decides which ones get to happen, not this.
MODEL_CALL_TIMEOUT_SECONDS: Final = 12.0

# One retrieval call, end to end — and "end to end" is the whole point of the arithmetic below.
#
# The spec's table says two seconds, which is `SEARCH_STATEMENT_TIMEOUT_MS`: the Postgres-side
# limit on the vector query alone. But a search is an *embedding round trip* and then that query,
# and P1 allows `EMBEDDING_TIMEOUT_SECONDS` for the first. Capping the pair at two seconds meant
# every live search was cancelled mid-embedding — measured, not guessed: three searches in three
# turns, all timing out at 2.0s with the embeddings response arriving afterwards, and nothing from
# the corpus ever reaching the model.
#
# Derived from P1's constants rather than written as a number, so the two cannot drift apart
# again. The extra second is slack for the connection itself.
#
# The cost is real: a slow search now spends a third of the turn budget. That is what the budget
# checks are for — `tools.py` will not start a search unless there is also time for the model call
# that would use its result.
RETRIEVAL_TIMEOUT_SECONDS: Final = (
    EMBEDDING_TIMEOUT_SECONDS + (SEARCH_STATEMENT_TIMEOUT_MS / 1000) + 1.0
)

# Under this much budget left, a phase is skipped rather than started. Starting a twelve-second
# call with one second left produces a timeout the user waits for and learns nothing from.
MIN_PHASE_SECONDS: Final = 2.0

# --- Round trips --------------------------------------------------------------------
#
# The model may search, see results, and search once more. After that the final call goes out
# with tools disabled, so it has to answer with what it has. More rounds is autonomy nobody
# specified, paid for in seconds the user is watching.
MAX_TOOL_ROUNDS: Final = 2

# Across both rounds. A model that is unsure repeats itself; five is enough for a genuine
# follow-up and short of a loop.
MAX_RETRIEVAL_CALLS: Final = 5

# Transient failures only — a timeout, a rate limit, a 5xx. A completed call with bad content is
# never retried blindly; that is the repair pass, and it carries the specific errors.
MAX_TRANSIENT_RETRIES: Final = 2

# Jittered exponential backoff between attempts. Jitter matters more than the base here: without
# it, every extension whose call failed at the same moment comes back at the same moment.
RETRY_BASE_SECONDS: Final = 0.4
RETRY_MAX_SECONDS: Final = 2.0

# One repair attempt per turn, whatever diagnosed it. A second is a conversation with a model
# about its own mistake, on the user's clock.
MAX_REPAIR_PASSES: Final = 1

# --- Quota --------------------------------------------------------------------------
#
# Fraction of the daily quota at which a user's approach to the limit is logged, so the limit
# being hit during a demo is something we saw coming rather than something we discover in the
# panel. The quota itself is `settings.daily_token_quota`.
QUOTA_WARN_FRACTION: Final = 0.8

# --- Circuit breaker ----------------------------------------------------------------
#
# Consecutive transport failures before the breaker opens. Three, because two can be one bad
# minute and a retry apiece.
BREAKER_FAILURE_THRESHOLD: Final = 3

# How long it stays open. Long enough that a provider outage is not paid for by every caller at
# twelve seconds each, short enough that a recovery is picked up while the user is still there.
BREAKER_OPEN_SECONDS: Final = 30.0

# --- The failure catalogue ----------------------------------------------------------

FailureCode = Literal[
    "MODEL_UNAVAILABLE",
    "MODEL_TIMEOUT",
    "BUDGET_EXCEEDED",
    "SCHEMA_FAILED",
    "SESSION_BUSY",
    "QUOTA_EXCEEDED",
]

# Printed verbatim in the side panel. Written for someone who has already lost a morning to a
# portal: what happened, and what to do now. No code, no jargon, no apology paragraph.
# HTTP status is deliberately not mapped here — P5 owns the endpoint's contract.
FAILURE_MESSAGES: Final[dict[FailureCode, str]] = {
    "MODEL_UNAVAILABLE": "I can't reach the assistant right now. Please try again in a moment.",
    "MODEL_TIMEOUT": "The assistant is taking too long. Try again in a moment.",
    "BUDGET_EXCEEDED": (
        "That took longer than I can wait. Try again, or ask me about one field at a time."
    ),
    "SCHEMA_FAILED": (
        "I couldn't put that into a reliable answer. Please ask me again, in fewer words."
    ),
    "SESSION_BUSY": "I'm still working on your last message. Give me a moment and try again.",
    "QUOTA_EXCEEDED": (
        "You've used up today's assistance on this account. Please try again tomorrow."
    ),
}

# What telemetry records for a turn that finished. Anything else in that field is a failure code.
OUTCOME_OK: Final = "ok"

# --- Tool messages ------------------------------------------------------------------
#
# What the model is told when a tool call cannot be executed. Each one ends by telling it not to
# answer from its own knowledge, because a model that has filled its context with page text
# reads the nearest instruction — and "the search failed" must never become "there is no rule".
SEARCH_LIMIT_MESSAGE: Final = (
    "Search limit reached for this turn. Answer with the sources you already have. If they do "
    "not cover the question, say you do not have official guidance on it. Do not answer from "
    "your own knowledge."
)

UNKNOWN_TOOL_MESSAGE: Final = (
    "That tool does not exist. The only tool available is the official-information search."
)

BAD_ARGUMENTS_MESSAGE: Final = (
    "Those arguments were not usable, so nothing was searched. Call the search again with a "
    "non-empty 'query' string, or answer without it and say you have no official guidance."
)

# --- Cost -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRate:
    """USD per million tokens, input and output, for one model."""

    input_per_million: float
    output_per_million: float


# NOT SET. Both figures are zero for every model, so `estimated_cost_usd` in the telemetry reads
# 0.0 until someone fills them in from the provider's current price list. Zero rather than a
# remembered number on purpose: a plausible-looking cost per turn is the kind of figure that ends
# up in a slide, and being wrong about it is worse than not having it. The token counts beside it
# are real, so the cost can be worked out by hand in the meantime.
#
# To set: put today's date and the source in this comment, and the two numbers below.
MODEL_RATES: Final[dict[str, ModelRate]] = {
    "gpt-4.1-mini": ModelRate(0.0, 0.0),
    "gpt-4.1": ModelRate(0.0, 0.0),
    "gpt-4o-mini": ModelRate(0.0, 0.0),
}

# What an unlisted model costs: nothing, reported as nothing. A model name absent from the table
# is logged once per turn so a deployment on something unpriced is visible in the log.
UNKNOWN_MODEL_RATE: Final = ModelRate(0.0, 0.0)

TOKENS_PER_MILLION: Final = 1_000_000
