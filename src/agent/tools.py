"""
The tool executor: retrieval, and nothing else.

One tool is offered to the model, and this module is the whole of what happens when it calls it.
The rules below all come from the same place — a tool the model can call is a capability the guard
has to cover, so the surface stays one function and the arguments are checked before anything runs:

- **Malformed arguments are answered, not raised.** A model that sent nonsense gets a tool message
  telling it so and keeps its turn; an exception would end a turn the user is watching over a
  missing quote.
- **Five retrieval calls per turn**, across both rounds. Past that the model is told the limit is
  reached, and nothing is executed.
- **Identical queries are served from a per-turn cache.** Models repeat themselves when unsure,
  and the same query costs an embedding round trip every time.
- **Empty is empty.** `documents/tools.py` builds the payload, and the three outcomes it keeps
  apart — chunks found, nothing found, lookup unavailable — are the reason the model does not fill
  a gap from memory. Nothing here loosens a query and searches again.
- **Every chunk is accumulated**, deduplicated by `chunk_id`, and handed to the guard. That list
  is what citation verification checks against, so a chunk missing from it becomes a dropped
  citation.

One deliberate difference from the spec's table, which reads "retrieval call 2s → treated as
empty": a timeout here is reported to the model as **unavailable**, not as empty. "Empty" means
"no official source covers this", which the model is instructed to relay as fact — and a slow
database is not evidence that a rule does not exist. `available: false` says the lookup did not
run, which is what actually happened.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.agent.calls import remaining_seconds
from src.agent.constants import (
    MAX_RETRIEVAL_CALLS,
    MIN_PHASE_SECONDS,
    RETRIEVAL_TIMEOUT_SECONDS,
    SEARCH_LIMIT_MESSAGE,
    UNKNOWN_TOOL_MESSAGE,
)
from src.agent.telemetry import TurnState, phase
from src.agent.tool_args import SearchArgs, parse_search_args
from src.ai.client import Message, ToolCallRequest, tool_result_message
from src.documents.schemas import RetrievalResult, RetrievedChunk
from src.documents.tools import SEARCH_TOOL_NAME, result_to_tool_payload

logger = logging.getLogger(__name__)

# `search_government_information`'s shape, as the runner needs it. Injected so a turn can be run
# against a stub with no database, which is how every check in this task runs.
RetrieveCallable = Callable[..., Awaitable[RetrievalResult]]


@dataclass
class ToolRun:
    """
    Per-turn tool state: how many searches have run, what they returned, and the cache.

    One instance per turn and never shared. The cache in particular must not outlive the turn —
    `documents/embeddings.py` already caches embeddings across requests with a TTL, and caching
    *results* globally would serve one turn's evidence to another turn's guard.
    """

    served: int = 0
    cache: dict[tuple[str, int, str, str], dict[str, Any]] = field(default_factory=dict)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    seen_chunk_ids: set[int] = field(default_factory=set)

    def accumulate(self, chunks: Sequence[RetrievedChunk]) -> None:
        """Keep every distinct chunk this turn retrieved, in the order it first arrived."""
        for chunk in chunks:
            if chunk.chunk_id in self.seen_chunk_ids:
                continue

            self.seen_chunk_ids.add(chunk.chunk_id)
            self.chunks.append(chunk)


def _unavailable_payload(note: str) -> dict[str, Any]:
    """
    The "lookup did not run" shape, with a specific note.

    Same three keys as `result_to_tool_payload`'s unavailable case, so the model never has to
    learn a second result shape, and `available: false` keeps "we could not check" distinct from
    "there is nothing to find".
    """
    return {"available": False, "found": False, "chunks": [], "note": note}


async def _run_search(
    args: SearchArgs,
    *,
    retrieve: RetrieveCallable,
    run: ToolRun,
    state: TurnState,
    deadline: float,
) -> dict[str, Any]:
    """One search, bounded, unable to fail the turn, and the only place chunks are kept."""
    remaining = remaining_seconds(deadline)

    # A search only earns its two seconds if there is still a model call left to use the result
    # in. Under the two together, the turn skips ahead: retrieving evidence for an answer that
    # can no longer be written spends the user's last seconds on nothing.
    if remaining < RETRIEVAL_TIMEOUT_SECONDS + MIN_PHASE_SECONDS:
        state.retrieval_unavailable += 1

        return _unavailable_payload(
            "There was no time left to check the official sources. Tell the user you could "
            "not check, and do not answer the question from your own knowledge.",
        )

    budget = min(RETRIEVAL_TIMEOUT_SECONDS, remaining)

    try:
        async with asyncio.timeout(budget):
            result = await retrieve(
                args.query,
                limit=args.limit,
                agency=args.agency,
                service=args.service,
            )
    except TimeoutError:
        logger.warning("retrieval timed out after %.1fs", budget)
        state.retrieval_unavailable += 1

        return _unavailable_payload(
            "The official-information lookup timed out. Tell the user you could not check the "
            "official sources right now, and do not answer from your own knowledge.",
        )
    except Exception as exc:  # noqa: BLE001  # a turn must survive any retrieval failure
        # Type only: the message can carry the query, and the query is what a citizen asked.
        logger.warning("retrieval failed (%s)", type(exc).__name__)
        state.retrieval_unavailable += 1

        return _unavailable_payload(
            "The official-information lookup could not be completed. Tell the user you could "
            "not check the official sources right now, and do not answer from your own "
            "knowledge.",
        )

    if not result.available:
        state.retrieval_unavailable += 1

    # Accumulated here rather than by the caller, because this is the one place a chunk exists
    # before it is flattened into a payload — and a chunk the guard never sees is a citation the
    # guard will drop, however real the source was.
    run.accumulate(result.chunks)
    state.chunks = len(run.chunks)

    return result_to_tool_payload(result)


async def execute_tool_calls(
    requests: Sequence[ToolCallRequest],
    *,
    retrieve: RetrieveCallable,
    run: ToolRun,
    state: TurnState,
    deadline: float,
) -> list[Message]:
    """
    Execute one round of tool calls and return the messages that answer them.

    Every request gets exactly one reply message with its own `tool_call_id`, in order — the
    provider requires it, and a missing one makes the next call fail as a malformed conversation.
    A refusal is still a reply.
    """
    messages: list[Message] = []

    with phase(state, "tools"):
        for request in requests:
            state.tool_calls += 1
            payload = await _execute_one(
                request,
                retrieve=retrieve,
                run=run,
                state=state,
                deadline=deadline,
            )
            messages.append(tool_result_message(request.id, payload))

    return messages


async def _execute_one(
    request: ToolCallRequest,
    *,
    retrieve: RetrieveCallable,
    run: ToolRun,
    state: TurnState,
    deadline: float,
) -> dict[str, Any]:
    """One tool call: name, arguments, caps, cache, execution."""
    if request.name != SEARCH_TOOL_NAME:
        logger.warning("model called an unknown tool (name=%s)", request.name)
        state.tool_errors += 1

        return _unavailable_payload(UNKNOWN_TOOL_MESSAGE)

    args = parse_search_args(request.arguments)

    if isinstance(args, str):
        # The model's own mistake, answered in its own turn. The reason is our sentence, not
        # anything the model wrote, so nothing untrusted is echoed back.
        state.tool_errors += 1

        return _unavailable_payload(args)

    cached = run.cache.get(args.cache_key)

    if cached is not None:
        logger.info("retrieval served from the turn cache")

        return cached

    # The cap counts calls that were *served*, cached or executed, not just those that reached
    # the database: it is there to bound how much a turn can grow, and a served result grows the
    # conversation whether or not it cost a round trip.
    if run.served >= MAX_RETRIEVAL_CALLS:
        logger.info("retrieval refused: %d calls already served this turn", run.served)

        return _unavailable_payload(SEARCH_LIMIT_MESSAGE)

    payload = await _run_search(
        args,
        retrieve=retrieve,
        run=run,
        state=state,
        deadline=deadline,
    )
    run.served += 1

    if payload.get("available") and payload.get("found"):
        # Only a real result is cached. Re-asking after a timeout is a reasonable thing for the
        # model to do, and serving it the timeout again from a cache would make that pointless.
        run.cache[args.cache_key] = payload

    return payload
