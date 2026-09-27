"""
The call sequence: first call, at most one more tool round, final call, parse.

This is steps 3 to 6 of the spec's sequence, and it is shared by both entry points — a plan turn
and an explain turn differ only in their system prompt and their schema, which is why the explain
turn is a real second caller rather than a copy.

Two decisions worth knowing:

**The schema is enforced on every call, not only the last one.** The spec's step 5 says that if an
earlier call already returned a schema-valid answer with no tool calls, that *is* the final call
and another must not be made. That is only possible if the earlier call carried the schema too —
so it does, and the common path (a question needing no search) costs one model call instead of
two. Tools are what changes between calls: offered for the first two, withdrawn for the last, so
the last call has no way to do anything but answer.

**The conversation never contains the model's answer.** Tool calls and tool results are appended
as they happen; the answer is parsed and returned, never appended. So the message list that comes
back out is exactly what a repair should reuse: the same page, the same retrieved sources, and no
trace of the attempt that failed.
"""

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.agent.calls import call_model, has_budget
from src.agent.constants import MAX_TOOL_ROUNDS
from src.agent.exceptions import TurnAborted
from src.agent.repair import (
    error_signatures,
    repair_instruction,
    repair_messages,
    schema_errors,
)
from src.agent.telemetry import TurnState
from src.agent.tools import RetrieveCallable, ToolRun, execute_tool_calls
from src.ai.client import (
    Message,
    ModelClient,
    ModelReply,
    assistant_tool_calls_message,
    system_message,
    user_message,
)
from src.documents.tools import SEARCH_TOOL

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def parse_reply(
    reply: ModelReply,
    model: type[ResponseT],
    state: TurnState | None = None,
) -> tuple[ResponseT | None, list[str]]:
    """
    Parse one completion into the response model, or say why it could not be.

    A refusal and an empty completion are handled here rather than treated as exceptions: both
    are a call that succeeded and returned nothing usable, which is the same problem as text that
    will not parse, and the repair pass is the same answer to it.

    `state`, when given, collects the failure signatures for the telemetry — locations and pydantic
    type slugs only, never a message and never the text that failed.
    """
    if reply.refusal:
        logger.warning("model refused to answer (finish_reason=%s)", reply.finish_reason)

        if state is not None:
            state.schema_errors += ("(root):refusal",)

        return None, ["- the model declined to answer in the required format"]

    if not reply.text:
        logger.warning("model returned no text (finish_reason=%s)", reply.finish_reason)

        if state is not None:
            state.schema_errors += (f"(root):empty_{reply.finish_reason or 'unknown'}",)

        return None, ["- the answer was empty"]

    try:
        return model.model_validate_json(reply.text), []
    except ValidationError as exc:
        # Counts only. The errors go back to the model, which wrote the offending text; they do
        # not go into a log line, because a validation error can quote what it rejected.
        signatures = error_signatures(exc)

        if state is not None:
            state.schema_errors += signatures

        logger.warning(
            "model answer failed validation (errors=%d, finish_reason=%s, failed=%s)",
            exc.error_count(),
            reply.finish_reason,
            ",".join(signatures),
        )

        return None, schema_errors(exc)


async def converse(
    *,
    model: type[ResponseT],
    schema: dict[str, Any],
    schema_name: str,
    system_prompt: str,
    context_text: str,
    client: ModelClient,
    retrieve: RetrieveCallable,
    run: ToolRun,
    state: TurnState,
    deadline: float,
) -> tuple[ResponseT, list[Message]]:
    """
    Run the call sequence and return the parsed answer plus the conversation behind it.

    Raises `TurnAborted` with `SCHEMA_FAILED` when neither the first answer nor its one repair
    would parse, and whatever `call_model` raises — `BUDGET_EXCEEDED`, `MODEL_TIMEOUT`,
    `MODEL_UNAVAILABLE` — when the provider is the problem.
    """
    messages: list[Message] = [system_message(system_prompt), user_message(context_text)]
    rounds = 0
    tools_offered = True

    while True:
        # Tools are offered while rounds remain. On the final call they are withdrawn entirely,
        # which is what makes it final: the model has no move left except to answer.
        tools_offered = rounds < MAX_TOOL_ROUNDS
        tools: Sequence[dict[str, Any]] = (SEARCH_TOOL,) if tools_offered else ()

        reply = await call_model(
            client=client,
            state=state,
            messages=messages,
            deadline=deadline,
            tools=tools,
            response_schema=schema,
            schema_name=schema_name,
        )

        if not reply.tool_calls:
            if not tools_offered:
                break

            # A reply with no tool calls, made while tools were still on offer, is *usually* the
            # answer — that is the fast path this sequence exists to allow. But it can also be the
            # model narrating what it just read: a live turn produced prose here on every search,
            # because a model with a tool still available is entitled to send an ordinary message
            # and strict mode only binds when it has nothing else to do.
            #
            # So a parse failure at this point is not a failed answer. The tools come off and the
            # model is asked once more, which is the call that has always parsed. Spending the
            # single repair pass here would leave a real schema failure with nothing left — and
            # the final call costs exactly what the repair would have.
            #
            # The failure is still recorded in the telemetry, because `(root):json_invalid` here
            # is how this behaviour was found in the first place, and a rule broken mid-sequence
            # (valid JSON, wrong shape) is worth seeing too.
            candidate, _ = parse_reply(reply, model, state)

            if candidate is not None:
                return candidate, messages

            state.discarded_replies += 1
            logger.info("mid-sequence reply was not the answer; making the final call")
            rounds = MAX_TOOL_ROUNDS

            continue

        if not tools_offered:
            # Tools were withdrawn for this call and it asked for one anyway. Its tool calls are
            # ignored rather than executed: the alternative is a loop whose only exit is the
            # twenty-second budget, paid for by the person watching the panel. With no answer to
            # parse, this falls through to the repair pass.
            logger.warning("model requested a tool after tools were withdrawn; ignoring")

            break

        messages.append(assistant_tool_calls_message(reply.tool_calls))
        messages.extend(
            await execute_tool_calls(
                reply.tool_calls,
                retrieve=retrieve,
                run=run,
                state=state,
                deadline=deadline,
            ),
        )
        rounds += 1

    parsed, errors = parse_reply(reply, model, state)

    if parsed is not None:
        return parsed, messages

    if state.repair_ran or not has_budget(deadline):
        # Out of repairs or out of time. The diagnosis is still the schema, so that is what the
        # user is told about — not the clock, which is not the thing that went wrong.
        raise TurnAborted("SCHEMA_FAILED")

    state.repair_ran = True
    state.repair_reason = "schema"

    repaired = await call_model(
        client=client,
        state=state,
        messages=repair_messages(messages, repair_instruction("schema", errors=errors)),
        deadline=deadline,
        response_schema=schema,
        schema_name=schema_name,
    )
    parsed, _ = parse_reply(repaired, model, state)

    if parsed is None:
        raise TurnAborted("SCHEMA_FAILED")

    return parsed, messages
