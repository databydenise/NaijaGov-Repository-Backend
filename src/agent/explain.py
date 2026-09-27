"""
The explain turn: the same machinery, a smaller schema, no actions.

`/explain` describes one field and never writes to the page, so there is nothing for the guard's
action checks to do here. What still applies is grounding, and it applies identically: the reply
is checked against the chunks this turn retrieved, an unsupported claim earns the one repair, and
a claim still unsupported after it is replaced with the guard's fixed sentence rather than shown.

`/explain` retrieves *before* it calls the model — that is how it answers "no official guidance"
without spending a model call — so `chunks` arrives already retrieved and is seeded into the
turn's tool state. Citation verification checks everything the turn holds, whether it was handed
over at the start or searched for mid-turn; a chunk the runner never saw is a citation it drops,
however real the source.

`src/explain/` owns the rendering of `context`.
"""

import logging
from collections.abc import Sequence

from src.agent.calls import call_model, has_budget
from src.agent.exceptions import TurnAborted
from src.agent.locks import turn_lock
from src.agent.quota import check_quota, record_usage
from src.agent.repair import repair_instruction, repair_messages
from src.agent.schemas import ExplainOutcome, ExplainTurn
from src.agent.service import (
    EXPLAIN_SCHEMA_NAME,
    as_failure,
    new_state,
    turn_deadline_from,
)
from src.agent.telemetry import TurnState, build_telemetry, log_turn
from src.agent.tools import RetrieveCallable, ToolRun
from src.agent.turn import converse, parse_reply
from src.ai.client import Message, ModelClient
from src.ai.prompt_loader import SYSTEM_EXPLAIN
from src.ai.schemas import EXPLAIN_RESPONSE_SCHEMA, Citation, ExplainResponse
from src.ai.utils import estimate_tokens
from src.documents.schemas import RetrievedChunk
from src.guard.constants import UNVERIFIED_REPLY
from src.guard.grounding import grounding_verdict, verify_citations

logger = logging.getLogger(__name__)


async def run_explain_turn(  # noqa: PLR0913  # one keyword per input to the turn
    *,
    context: str,
    retrieve: RetrieveCallable,
    client: ModelClient,
    session_id: str,
    user_id: str,
    chunks: Sequence[RetrievedChunk] = (),
    deadline: float | None = None,
) -> ExplainOutcome:
    """
    Run one explain turn: the same machinery, a smaller schema, no actions.

    `/explain` describes a field and never writes to the page, so there is nothing for the guard's
    action checks to do. What still applies is grounding: the reply is checked against the chunks
    this turn holds, an unsupported claim earns the one repair, and a claim that is still
    unsupported afterwards is replaced with the guard's fixed sentence.

    `chunks` are the ones the caller already retrieved and rendered into `context`. They are seeded
    into the turn's tool state so a citation to one of them verifies: the caller's retrieval and
    the model's own search end up in the same list, and a citation is checked against the whole of
    it. The search tool is still offered, so a model whose sources fall short can look for more.
    """
    state = new_state("explain", client, session_id, user_id)
    state.context_tokens = estimate_tokens(context)
    turn_deadline = turn_deadline_from(deadline)
    run = ToolRun()
    run.accumulate(chunks)
    state.chunks = len(run.chunks)

    try:
        async with turn_lock(session_id):
            check_quota(user_id)

            response, messages = await converse(
                model=ExplainResponse,
                schema=EXPLAIN_RESPONSE_SCHEMA,
                schema_name=EXPLAIN_SCHEMA_NAME,
                system_prompt=SYSTEM_EXPLAIN,
                context_text=context,
                client=client,
                retrieve=retrieve,
                run=run,
                state=state,
                deadline=turn_deadline,
            )

            citations, verdict = _verify(response, run)

            if verdict == "unverified" and not state.repair_ran and has_budget(turn_deadline):
                response = await _repair_explain(
                    response,
                    messages=messages,
                    client=client,
                    state=state,
                    deadline=turn_deadline,
                )
                citations, verdict = _verify(response, run)

            if verdict == "unverified":
                # Second attempt, still unsupported — or no second attempt was possible. The
                # claim goes, and with nothing cited there is nothing to show beside it. The
                # example goes with it: "e.g. ₦5,000" is the same unsupported claim in fewer
                # words, and it is the part of the answer a user is most likely to copy.
                response = ExplainResponse(explanation=UNVERIFIED_REPLY, example=None, citations=[])
                citations = []
            else:
                # The model's own citation list is never passed through: `verify_citations`
                # rebuilds each one from the chunk it names, so a real-looking URL paired with a
                # claim that URL does not make cannot survive.
                response = ExplainResponse(
                    explanation=response.explanation,
                    example=response.example,
                    citations=list(citations),
                )

            record = build_telemetry(state, grounding=verdict)
            log_turn(record)

            return ExplainTurn(
                response=response,
                citations=tuple(citations),
                grounding=verdict,
                chunks=tuple(run.chunks),
                telemetry=record,
            )
    except TurnAborted as abort:
        return as_failure(abort, state)
    finally:
        record_usage(user_id, state.total_tokens)


def _verify(response: ExplainResponse, run: ToolRun) -> tuple[list[Citation], str]:
    """
    Verified citations and the grounding verdict for an explain reply.

    The example is scanned with the explanation rather than beside it. "e.g. ₦5,000" is a claim
    about a fee wherever it sits in the answer, and a marker check that read only the explanation
    would let the shortest, most copyable version of an invented fact through ungrounded.
    """
    citations, _ = verify_citations(response.citations, run.chunks)
    claimable = f"{response.explanation} {response.example or ''}"

    return citations, grounding_verdict(claimable, citations)


async def _repair_explain(
    response: ExplainResponse,
    *,
    messages: Sequence[Message],
    client: ModelClient,
    state: TurnState,
    deadline: float,
) -> ExplainResponse:
    """The one repair attempt for an explain reply. Returns the repaired answer, or the original."""
    state.repair_ran = True
    state.repair_reason = "unverified"

    try:
        reply = await call_model(
            client=client,
            state=state,
            messages=repair_messages(messages, repair_instruction("unverified")),
            deadline=deadline,
            response_schema=EXPLAIN_RESPONSE_SCHEMA,
            schema_name=EXPLAIN_SCHEMA_NAME,
        )
    except TurnAborted as abort:
        logger.info("explain repair abandoned (%s); the claim will be replaced", abort.code)

        return response

    repaired, _ = parse_reply(reply, ExplainResponse, state)

    return repaired if repaired is not None else response
