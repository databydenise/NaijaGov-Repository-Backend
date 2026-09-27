"""
The `/explain` sequence: short-circuit, look up, retrieve, run, cache.

Read this file as a series of chances not to call a model, because that is what it is. Four of the
five outcomes cost nothing:

1. **A sensitive field** answers from fixed copy. No lookup, no model, no cache.
2. **A repeat click** answers from the cache table, shared across users.
3. **A corpus that does not cover the field** answers with the no-guidance sentence. The honest
   answer is the cheap one, which is the point of doing retrieval before the model call rather
   than handing the model a search tool and hoping.
4. **A lookup that did not run** answers with a different sentence, because "no official source
   covers this" is a claim about the corpus and a database that timed out is not evidence for it.
5. **Only a genuine hit** reaches `agent/explain.py`.

Nothing here writes to the session. No history entry, no chat values, no plan, no `updated_at`
touched: `/explain` is a side call from READY, and a stale page produces a slightly stale
explanation rather than a wrong fill — which is why, unlike `/plan`, there is no page hash to
check and no `SESSION_NOT_FOUND` to raise.
"""

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.constants import TURN_BUDGET_SECONDS
from src.agent.explain import run_explain_turn
from src.agent.schemas import ExplainTurn, TurnFailure
from src.ai.client import shared_client
from src.ai.context import render_explain_context
from src.ai.prompt_loader import EXPLAIN_PROMPT_VERSION
from src.auth.schemas import AuthedUser
from src.documents.schemas import RetrievalResult
from src.documents.service import search_government_information
from src.explain import metrics, store
from src.explain import utils as explain_utils
from src.explain.constants import (
    CACHE_RESERVE_SECONDS,
    NO_GUIDANCE_EXPLANATION,
    REQUEST_BUDGET_SECONDS,
    RETRIEVAL_UNAVAILABLE_EXPLANATION,
    SEARCH_LIMIT,
)
from src.explain.scope import corpus_stamp, resolve_scope, step_view
from src.explain.schemas import ExplainRequest, ExplainResponse, ExplainScope
from src.turn_errors import turn_failed

logger = logging.getLogger(__name__)


def _turn_deadline(started: float) -> float:
    """
    When the turn must be finished, on the monotonic clock.

    The tighter of two bounds: the runner's own budget from now, and what is left of the
    request's budget once time is set aside for the cache write. The second only bites when the
    retrieval before the turn was slow — which is exactly when a full-length turn would otherwise
    become a request the panel has already given up waiting for.
    """
    runner_bound = time.monotonic() + TURN_BUDGET_SECONDS
    request_bound = started + REQUEST_BUDGET_SECONDS - CACHE_RESERVE_SECONDS

    return min(runner_bound, request_bound)


async def _retrieve(scope: ExplainScope, query: str) -> RetrievalResult:
    """
    The one search this endpoint makes, scoped to the workflow's own agency and service.

    Filtered rather than corpus-wide because the question is about a field on *this* agency's form,
    and an FRSC answer to a question about a CAC form is worse than no answer. The filters are exact
    matches on what the registry holds, so a registry and a corpus that do not share vocabulary
    return nothing — which surfaces as "no official guidance" rather than as a wrong source, and
    is a content problem to fix in the seed rather than a filter to loosen here.
    """
    return await search_government_information(
        query,
        limit=SEARCH_LIMIT,
        agency=scope.agency,
        service=scope.service,
    )


async def _run_turn(
    payload: ExplainRequest,
    *,
    scope: ExplainScope,
    result: RetrievalResult,
    user_id: uuid.UUID,
    started: float,
) -> ExplainTurn | TurnFailure:
    """
    Render the context and run the turn. No database work, and nothing is written.

    The retrieved chunks go in twice, and both are needed: into the prompt, where the model reads
    them, and into the turn, where a citation is checked against them. The lock key is this field
    on this session rather than the session itself — a plan turn in flight must not stop the user
    reading about a field, and two clicks on the *same* field are the same question asked twice.
    """
    context = render_explain_context(
        step=step_view(scope),
        page_field=payload.field,
        chunks=result.chunks,
        nearby_text=payload.nearby_text,
        question=payload.question or "",
    )

    return await run_explain_turn(
        context=context.text,
        retrieve=search_government_information,
        client=shared_client(),
        session_id=_turn_key(payload, user_id),
        user_id=str(user_id),
        chunks=tuple(result.chunks),
        deadline=_turn_deadline(started),
    )


def _turn_key(payload: ExplainRequest, user_id: uuid.UUID) -> str:
    """
    The runner's lock and telemetry key: one in-flight explain per field.

    Prefixed, so it can never collide with the plain session id `/plan` locks on. An explain turn
    and a plan turn are not two plans racing for one page — which is what that lock exists to
    prevent — and refusing to explain a field because a plan is still running would be a worse
    panel for no benefit the quota check does not already provide.
    """
    scope = payload.session_id or user_id

    return f"explain:{scope}:{payload.field.field_id}"


async def explain_field(
    db: AsyncSession,
    user: AuthedUser,
    payload: ExplainRequest,
) -> ExplainResponse:
    """
    Explain one field: two or three sentences, an example where one is obvious, and a source.

    Raises an error-contract `HTTPException` only for a turn that could not finish. Retrieval
    finding nothing and an unknown session are both normal answers with a 200, as the spec's error
    table is careful to say: neither is a failure of this request, and both are things a citizen
    needs told rather than hidden behind an error code.
    """
    started = time.monotonic()
    page_field = payload.field

    kind = explain_utils.sensitive_kind(page_field)

    if kind is not None:
        # The one answer that is better for being fixed. No lookup, no model call, no cache row:
        # this is the sentence we want on stage at the moment the Copilot visibly declines to act.
        return explain_utils.fixed_answer(
            page_field.field_id,
            explain_utils.canned_explanation(kind),
            grounded=True,
        )

    scope = await resolve_scope(db, payload.session_id, user.id)
    key = store.cache_key_for(
        scope,
        page_field.label,
        prompt_version=EXPLAIN_PROMPT_VERSION,
        corpus_stamp=await corpus_stamp(db, scope),
        has_question=bool(payload.question),
    )

    if key is not None:
        row = await store.get_cached(key)
        metrics.record_lookup(hit=row is not None)

        if row is not None:
            return explain_utils.cached_answer(page_field.field_id, row)

    query = explain_utils.build_query(page_field, payload.nearby_text, payload.question)
    result = await _retrieve(scope, query)

    if not result.available:
        # The lookup did not run. Never reported as "no official source covers this": that is a
        # claim about the corpus, and this is a claim about our database.
        return explain_utils.fixed_answer(
            page_field.field_id,
            RETRIEVAL_UNAVAILABLE_EXPLANATION,
            grounded=False,
        )

    if result.is_empty:
        return explain_utils.fixed_answer(
            page_field.field_id,
            NO_GUIDANCE_EXPLANATION,
            grounded=False,
        )

    outcome = await _run_turn(
        payload,
        scope=scope,
        result=result,
        user_id=user.id,
        started=started,
    )

    if isinstance(outcome, TurnFailure):
        raise turn_failed(outcome)

    response = explain_utils.answer_from_turn(page_field.field_id, outcome)

    if key is not None and response.grounded:
        # Grounded answers only, and never a custom question's — `key` is None for one. An
        # ungrounded answer is the one most worth retrying after the corpus grows, and a week-long
        # row would make that retry impossible.
        await store.put_cached(
            key,
            explanation=response.explanation,
            example=response.example,
            sources=response.sources,
            prompt_version=EXPLAIN_PROMPT_VERSION,
        )

    return response
