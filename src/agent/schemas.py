"""
What a turn returns.

Frozen dataclasses, like `guard/schemas.py` and for the same reason: nothing here parses
untrusted input — the model's output has already been through `PlanResponse` and the guard — and
these are built on the `/plan` path. P5 maps them onto its response model.

Three results, and a turn returns exactly one of them:

- `PlanTurn` — a guarded plan, its grounding verdict, the chunks the guard checked citations
  against, and the telemetry record.
- `ExplainTurn` — the same shape for the smaller explain schema.
- `TurnFailure` — a code, a sentence the panel prints verbatim, and telemetry all the same,
  because a turn that failed is exactly the turn worth measuring.

`TurnTelemetry` is content-free by construction: every field is a name, a count, a duration or a
verdict. There is no field for the message, the reply, a retrieved chunk, a resolved value, or a
label, and adding one is a privacy decision rather than an observability improvement.
"""

from dataclasses import dataclass, field

from src.agent.constants import FailureCode
from src.ai.schemas import Citation, ExplainResponse
from src.documents.schemas import RetrievedChunk
from src.guard.constants import GroundingVerdict
from src.guard.schemas import GuardedPlan


@dataclass(frozen=True)
class TurnTelemetry:
    """
    One content-free record per turn, for the log line and for tuning.

    `outcome` is `"ok"` or a failure code. Session and user ids are here because they are ids —
    they identify a row, not a person's data, and without them one slow turn cannot be told from
    a user having a slow day.
    """

    kind: str
    outcome: str
    model: str
    prompt_version: str
    session_id: str
    user_id: str

    duration_ms: float
    phase_ms: dict[str, float] = field(default_factory=dict)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    model_calls: int = 0
    retries: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    retrieval_unavailable: int = 0
    chunks_retrieved: int = 0

    repair_ran: bool = False
    repair_reason: str | None = None

    # Mid-sequence replies that were not the answer. One extra model call each.
    discarded_replies: int = 0

    # `location:type` for each validation rule the model broke, e.g. `actions.0:value_error`.
    # Our own field paths and pydantic's own slugs, so this stays content-free.
    schema_errors: tuple[str, ...] = ()

    grounding: str = "not_required"
    approved: int = 0
    rejected: int = 0

    context_tokens: int = 0
    truncated: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class PlanTurn:
    """A finished plan turn. `chunks` is every chunk retrieved, in the order it arrived."""

    plan: GuardedPlan
    grounding: GroundingVerdict
    chunks: tuple[RetrievedChunk, ...]
    telemetry: TurnTelemetry

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class ExplainTurn:
    """
    A finished explain turn.

    `response` carries the reply the user sees and the citations that survived verification — the
    model's own citation list is never passed through. An ungrounded reply that a repair could not
    fix has already been replaced with the guard's fixed sentence by the time it gets here.
    """

    response: ExplainResponse
    citations: tuple[Citation, ...]
    grounding: GroundingVerdict
    chunks: tuple[RetrievedChunk, ...]
    telemetry: TurnTelemetry

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class TurnFailure:
    """
    A turn that could not finish, with the sentence the panel prints verbatim.

    Returned, never raised: a failed turn is a normal outcome with a normal response, and a
    caller that has to catch something will eventually not.
    """

    code: FailureCode
    message: str
    telemetry: TurnTelemetry

    @property
    def ok(self) -> bool:
        return False


# What either entry point returns. A caller checks `.ok` or matches on the type.
PlanOutcome = PlanTurn | TurnFailure
ExplainOutcome = ExplainTurn | TurnFailure
