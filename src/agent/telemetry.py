"""
The turn's measurements, and the one log line they become.

`TurnState` is the mutable accumulator every phase writes to as it goes; `TurnTelemetry` is the
frozen record built from it at the end. They are separate because the log line must be emitted
for a failed turn too, and a turn can fail anywhere — so the numbers are collected as they
happen rather than assembled from a result that may not exist.

The rule that shapes this module: **nothing here can carry content.** No field takes a string a
user or a model wrote. What goes in is a model name, a phase duration, a token count, a verdict,
a code. That is what makes the telemetry safe to keep, which is the point of having it — it is
what a prompt change or a threshold sweep is judged against.

Cost is an estimate from `MODEL_RATES`, which is not filled in yet (see `constants.py`). It reads
0.0 until it is, deliberately.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.agent.constants import (
    MODEL_RATES,
    OUTCOME_OK,
    TOKENS_PER_MILLION,
    UNKNOWN_MODEL_RATE,
)
from src.agent.schemas import TurnTelemetry
from src.ai.client import ModelReply
from src.ai.prompt_loader import PROMPT_VERSION

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """
    Everything measured about a turn in progress. Mutable, single-turn, never shared.

    `kind` is `"plan"` or `"explain"`; the two share this machinery and are told apart in the log
    by this field alone.
    """

    kind: str
    model: str
    session_id: str
    user_id: str

    started: float = field(default_factory=time.monotonic)
    phases: dict[str, float] = field(default_factory=dict)

    model_calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    tool_calls: int = 0
    tool_errors: int = 0
    retrieval_unavailable: int = 0
    chunks: int = 0

    repair_ran: bool = False
    repair_reason: str | None = None

    # Replies that came back mid-sequence and were not the answer — the model narrating what it
    # just read rather than answering. Each one costs a model call, so this is the number to watch
    # if the tool rounds are ever reconsidered.
    discarded_replies: int = 0

    # Which of our own validation rules the model's answer broke, as `location:type`. Content-free
    # by construction — see `repair.error_signatures`.
    schema_errors: tuple[str, ...] = ()

    context_tokens: int = 0
    truncated: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_reply(self, reply: ModelReply) -> None:
        """Count one completed model call and the tokens it cost."""
        self.model_calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens

    def elapsed(self) -> float:
        return time.monotonic() - self.started


@contextmanager
def phase(state: TurnState, name: str) -> Iterator[None]:
    """
    Time one phase into `state.phases`, accumulating across visits.

    Three names are used — `model`, `tools`, `guard` — so a slow turn can be attributed without
    another timer. Accumulating rather than overwriting matters because a turn makes up to four
    model calls and the interesting number is the total.
    """
    started = time.perf_counter()

    try:
        yield
    finally:
        state.phases[name] = state.phases.get(name, 0.0) + (time.perf_counter() - started) * 1000


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    The turn's cost in USD, from the rate table.

    Returns 0.0 for every model until `MODEL_RATES` is filled in. An unpriced model name is
    logged once per turn, so a deployment on something the table does not know about is visible
    rather than silently free.
    """
    rate = MODEL_RATES.get(model)

    if rate is None:
        logger.warning("no cost rate for model %s; reporting 0.0", model)
        rate = UNKNOWN_MODEL_RATE

    input_cost = prompt_tokens * rate.input_per_million / TOKENS_PER_MILLION
    output_cost = completion_tokens * rate.output_per_million / TOKENS_PER_MILLION

    return round(input_cost + output_cost, 6)


def build_telemetry(
    state: TurnState,
    *,
    outcome: str = OUTCOME_OK,
    grounding: str = "not_required",
    approved: int = 0,
    rejected: int = 0,
) -> TurnTelemetry:
    """The frozen record for this turn. Called once, on the way out, success or failure."""
    return TurnTelemetry(
        kind=state.kind,
        outcome=outcome,
        model=state.model,
        prompt_version=PROMPT_VERSION,
        session_id=state.session_id,
        user_id=state.user_id,
        duration_ms=round(state.elapsed() * 1000, 1),
        phase_ms={name: round(value, 1) for name, value in state.phases.items()},
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        estimated_cost_usd=estimate_cost(
            state.model,
            state.prompt_tokens,
            state.completion_tokens,
        ),
        model_calls=state.model_calls,
        retries=state.retries,
        tool_calls=state.tool_calls,
        tool_errors=state.tool_errors,
        retrieval_unavailable=state.retrieval_unavailable,
        chunks_retrieved=state.chunks,
        repair_ran=state.repair_ran,
        repair_reason=state.repair_reason,
        discarded_replies=state.discarded_replies,
        schema_errors=state.schema_errors,
        grounding=grounding,
        approved=approved,
        rejected=rejected,
        context_tokens=state.context_tokens,
        truncated=state.truncated,
    )


def log_turn(record: TurnTelemetry) -> None:
    """
    One line per turn, as `key=value` pairs.

    Flat text rather than JSON because the rest of this service logs flat text; when structured
    logging is chosen, this is the one function that changes. Every value below is a number, an
    id, or a word from a fixed set — there is nothing to redact, by construction.
    """
    logger.info(
        "turn kind=%s outcome=%s model=%s prompt_version=%s session_id=%s user_id=%s "
        "duration_ms=%.0f phases=%s prompt_tokens=%d completion_tokens=%d cost_usd=%.6f "
        "model_calls=%d retries=%d tool_calls=%d tool_errors=%d retrieval_unavailable=%d "
        "chunks=%d repair=%s repair_reason=%s discarded=%d schema_errors=%s grounding=%s "
        "approved=%d rejected=%d context_tokens=%d truncated=%s",
        record.kind,
        record.outcome,
        record.model,
        record.prompt_version,
        record.session_id,
        record.user_id,
        record.duration_ms,
        ",".join(f"{name}:{value:.0f}" for name, value in sorted(record.phase_ms.items())),
        record.prompt_tokens,
        record.completion_tokens,
        record.estimated_cost_usd,
        record.model_calls,
        record.retries,
        record.tool_calls,
        record.tool_errors,
        record.retrieval_unavailable,
        record.chunks_retrieved,
        record.repair_ran,
        record.repair_reason or "none",
        record.discarded_replies,
        ",".join(record.schema_errors) or "none",
        record.grounding,
        record.approved,
        record.rejected,
        record.context_tokens,
        ",".join(record.truncated) or "none",
    )
