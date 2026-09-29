"""
The `/plan` sequence: load, check, run, persist, answer.

Every hard part of this endpoint happens somewhere else — P1 retrieves, P2 renders, P3 guards, P4
runs the turn under its budgets. This module is the orchestration around them, and its whole job
is to get four things right:

- **The page is the page.** The hash is recomputed from the snapshot and compared with the
  session's before anything else happens. A plan built against a snapshot of a different page is
  the worst outcome available here, so a mismatch is a refusal and never a best effort.
- **An identical retry costs nothing.** Same session, same message, same page, inside the window:
  the stored answer comes back. A flaky connection retrying should not cost the user seconds or
  the project money.
- **Nothing is persisted unless the turn produced something.** A failed turn raises, the request's
  transaction rolls back, and the session is exactly as it was. This is why the two database
  writes sit on either side of the turn and never inside it.
- **Values given in chat go into the session, never the profile.** There is no consent screen
  behind a chat message. They live as long as the session and die with it.

The persistence step stores what the *guard* resolved, not what the model said: the same values
the user is about to see in the preview.
"""

import logging
import time
import uuid
from collections.abc import Mapping, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.constants import TURN_BUDGET_SECONDS
from src.agent.plan import run_plan_turn
from src.agent.schemas import PlanTurn, TurnFailure
from src.ai.client import shared_client
from src.ai.context import (
    StepView,
    WorkflowStepLine,
    WorkflowView,
    render_plan_context,
)
from src.auth.schemas import AuthedUser
from src.context.utils import compute_page_hash
from src.documents.service import search_government_information
from src.plan import store
from src.plan import utils as plan_utils
from src.plan.constants import (
    PERSIST_RESERVE_SECONDS,
    REQUEST_BUDGET_SECONDS,
    UNKNOWN_STEP_NAME,
)
from src.plan.exceptions import page_changed, session_not_found, turn_failed
from src.plan.schemas import PlanRequest, PlanResponse, StepOut
from src.plan.utils import blocked_field_ids, build_response
from src.profiles import service as profiles_service
from src.profiles.constants import PROFILE_FIELDS
from src.sessions import service as sessions_service
from src.sessions.models import Session
from src.workflows import service as workflows_service
from src.workflows.schemas import Step

logger = logging.getLogger(__name__)


def _turn_deadline(started: float) -> float:
    """
    When the turn must be finished, on the monotonic clock.

    The tighter of two bounds: the runner's own budget from now, and what is left of the request's
    budget once time is set aside for the writes that follow. The second only bites when the
    reads before the turn were slow — which is exactly when a 20-second turn would otherwise
    become a request the panel has already given up waiting for.
    """
    runner_bound = time.monotonic() + TURN_BUDGET_SECONDS
    request_bound = started + REQUEST_BUDGET_SECONDS - PERSIST_RESERVE_SECONDS

    return min(runner_bound, request_bound)


async def _profile_values(db: AsyncSession, user_id: uuid.UUID) -> dict[str, str | None]:
    """
    The profile as the renderer and the guard take it: `{profile key: value or None}`.

    Built from `PROFILE_FIELDS` rather than from the row's `__dict__`, so a column added to the
    table — `password_hash` being the one that matters — cannot become a key a `value_ref` is
    allowed to resolve against.
    """
    profile = await profiles_service.load_profile(db, user_id)

    return {key: getattr(profile, key, None) for key in PROFILE_FIELDS}


async def _workflow_view(
    db: AsyncSession,
    workflow_id: str,
    steps: Sequence[Step],
    current_index: int | None,
) -> WorkflowView | None:
    """
    The whole workflow as the prompt shows it, or None if the registry no longer holds it.

    The steps are already in hand — the caller read them to place the current page — so this adds
    one cached registry read for the workflow's own name and agency and no database work at all on
    a warm cache. That cheapness is the point: "what do I do next" is the most common question the
    panel gets and the one the page itself can least often answer, so it must not cost a query.
    """
    workflows = await workflows_service.list_active_workflows(db)
    workflow = next((item for item in workflows if item.id == workflow_id), None)

    if workflow is None:
        return None

    return WorkflowView(
        name=workflow.name,
        agency=workflow.agency,
        steps=tuple(
            WorkflowStepLine(
                name=item.name,
                index=item.index,
                is_final=item.is_final,
                field_labels=item.field_labels,
            )
            for item in steps
        ),
        current_index=current_index,
    )


async def _step_of(
    db: AsyncSession,
    session: Session,
) -> tuple[StepView, StepOut | None, WorkflowView | None]:
    """
    Where this page sits in its workflow, in the three shapes its callers need.

    One view for the prompt's step line, one for the response, and the whole workflow for the
    `<workflow>` block.

    A session with no workflow is normal, not broken — `/context` writes one for any page outside
    the registry and the panel still offers chat there. That page gets a step view that claims
    nothing about position, no `step` in the response and no workflow block, rather than a
    fabricated "step 1 of 1" for a workflow we have never seen.

    Both reads are served from the registry cache, so this is usually no database work at all.
    """
    if session.workflow_id is None or session.step_id is None:
        return StepView(name=UNKNOWN_STEP_NAME, index=1, total=1), None, None

    steps = await workflows_service.get_steps(db, session.workflow_id)
    step = next((item for item in steps if item.id == session.step_id), None)

    if step is None:
        # The workflow was re-seeded under the session's feet. The page is still readable, and the
        # workflow's own steps are still worth showing — only our claim about *which* of them the
        # user is on has gone stale, so the list is rendered with nothing marked.
        logger.info(
            "session step is no longer in the registry (workflow=%s step=%s)",
            session.workflow_id,
            session.step_id,
        )
        workflow = await _workflow_view(db, session.workflow_id, steps, None)

        return StepView(name=UNKNOWN_STEP_NAME, index=1, total=1), None, workflow

    workflow = await _workflow_view(db, session.workflow_id, steps, step.index)

    return (
        StepView(name=step.name, index=step.index, total=len(steps)),
        StepOut(id=step.id, name=step.name, index=step.index, total=len(steps)),
        workflow,
    )


async def _run_turn(  # noqa: PLR0913  # every argument is a distinct input to the turn
    *,
    payload: PlanRequest,
    session: Session,
    user_id: uuid.UUID,
    profile: Mapping[str, str | None],
    step_view: StepView,
    workflow: WorkflowView | None,
    started: float,
) -> PlanTurn | TurnFailure:
    """Render the context and run the turn. No database work, and nothing is written."""
    context = render_plan_context(
        step=step_view,
        fields=payload.fields,
        buttons=payload.buttons,
        links=payload.links,
        workflow=workflow,
        profile=profile,
        chat_values=session.chat_values,
        history=session.history,
        message=payload.message,
    )

    return await run_plan_turn(
        context=context,
        fields=payload.fields,
        buttons=payload.buttons,
        links=payload.links,
        profile=profile,
        chat_values=session.chat_values,
        blocked_field_ids=blocked_field_ids(payload),
        retrieve=search_government_information,
        client=shared_client(),
        session_id=str(session.id),
        user_id=str(user_id),
        deadline=_turn_deadline(started),
    )


async def _persist(
    db: AsyncSession,
    *,
    session: Session,
    payload: PlanRequest,
    outcome: PlanTurn,
) -> None:
    """
    Everything this turn leaves behind, in the request's one transaction.

    The history entry stores the reply the *user sees* — the guard's, after an ungrounded claim
    has been replaced — so the next turn's context cannot repeat a sentence we already withdrew.

    `action_log` takes the refusals only. An approved action has not happened yet; its execution
    status arrives with `/results`.
    """
    await sessions_service.record_turn(
        db,
        session,
        user_message=payload.message,
        reply=outcome.plan.reply,
        chat_values=outcome.plan.chat_values,
    )

    await sessions_service.record_actions(
        db,
        session.id,
        plan_utils.action_log_entries(outcome.plan),
    )


async def create_plan(
    db: AsyncSession,
    user: AuthedUser,
    payload: PlanRequest,
) -> PlanResponse:
    """
    Plan one turn for this page and this message.

    Raises an error-contract `HTTPException` for every failure — a lost session, a changed page,
    or a turn that could not finish — so a request that does not produce a plan also does not
    commit: the handler's session rolls back on the way out.
    """
    started = time.monotonic()
    page_hash = compute_page_hash(payload.url, payload.fields)

    session = await sessions_service.get_user_session(db, payload.session_id, user.id)

    if session is None:
        raise session_not_found()

    if session.page_hash != page_hash:
        logger.info("plan refused: page changed (session=%s)", session.id)

        raise page_changed()

    key = store.idempotency_key(session.id, page_hash, payload.message)
    answered = store.get_answer(key)

    if answered is not None:
        # The same question again: the stored plan, its own id intact, so an approval still
        # matches what is held for it. Only the echoed correlation id follows this request.
        return answered.model_copy(
            update={"cached": True, "client_plan_id": payload.client_plan_id},
        )

    profile = await _profile_values(db, user.id)
    step_view, step_out, workflow = await _step_of(db, session)

    outcome = await _run_turn(
        payload=payload,
        session=session,
        user_id=user.id,
        profile=profile,
        step_view=step_view,
        workflow=workflow,
        started=started,
    )

    if isinstance(outcome, TurnFailure):
        raise turn_failed(outcome)

    await _persist(db, session=session, payload=payload, outcome=outcome)

    plan_id = store.new_plan_id()
    response = build_response(
        plan_id=plan_id,
        payload=payload,
        outcome=outcome,
        step=step_out,
    )

    store.remember_plan(
        plan_id,
        session_id=session.id,
        user_id=user.id,
        actions=tuple(response.actions),
    )
    store.remember_answer(key, response)

    return response
