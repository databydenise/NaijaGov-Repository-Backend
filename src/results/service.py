"""
The `/results` sequence: identify, claim, write once, answer.

The endpoint is the smallest in the system and almost all of its difficulty is in one word from
the spec: *idempotent*. An extension retries on a flaky connection, so the same run can arrive
twice, and duplicate rows would silently corrupt every number built on this table. So the order
below is deliberate:

1. **Resolve the session and the plan.** Either being unknown is a normal answer, not an error —
   the extension is reporting history, not asking permission, and a user whose plan expired while
   they were typing an OTP should not see an error about our bookkeeping.
2. **Replay an existing report.** If this plan has a row, its stored answer comes back untouched.
3. **Claim the plan before writing anything.** The `plan_reports` insert goes first, conflicting
   on the plan id and doing nothing if it is already there. Two simultaneous reports therefore
   have exactly one winner, and the loser replays rather than writing a second set of rows.
4. **Then write.** Action-log rows, the checkpoint event, the session's counters, the durable
   counters — all in the request's one transaction, the same one the claim was made in.

The claim being in that transaction is what makes the order safe rather than merely hopeful: a
failure anywhere after it rolls the claim back with everything else, so the report is retryable
and nothing is half-written. What the claim defends against is the *other* request — two reports
of one plan arriving together, where the second one's insert returns no row and it replays instead
of writing a second set of rows. A duplicated report is a wrong number nothing downstream can
detect; a retryable one is not a problem at all.
"""

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.schemas import AuthedUser
from src.plan import store as plan_store
from src.plan.schemas import PlannedActionOut
from src.results import utils as results_utils
from src.results.constants import DEFAULT_HINT_CODE
from src.results.models import CheckpointEvent, PlanReport, ResultsCounter
from src.results.schemas import (
    ResultsRequest,
    ResultsResponse,
    RunTotals,
    StepOut,
)
from src.sessions import service as sessions_service
from src.sessions.models import Session
from src.workflows import service as workflows_service

logger = logging.getLogger(__name__)


def _unacknowledged(step: StepOut | None = None) -> ResultsResponse:
    """
    The answer when there is no plan to record against.

    Same shape as a recorded run, so the panel renders one summary card and never branches on
    whether this service happened to still know about the run. The hint claims nothing about the
    page, because with no plan we know nothing about it — and "review it, then click Continue
    yourself" is true whatever happened.
    """
    return ResultsResponse(
        acknowledged=False,
        recorded=0,
        next_hint=results_utils.next_hint(DEFAULT_HINT_CODE),
        step=step,
        run=RunTotals(),
    )


async def _step_of(db: AsyncSession, session: Session) -> StepOut | None:
    """
    Where this run happened, or None for a page the registry does not know.

    `is_final` is the only part the hint depends on, and it is the reason this read happens at all:
    telling someone to submit a form themselves is only right on the last step. A session with no
    workflow gets no step rather than an invented "1 of 1" — the same call `/plan` makes.

    Served from the registry's five-minute cache, so this is usually no database work.
    """
    if session.workflow_id is None or session.step_id is None:
        return None

    steps = await workflows_service.get_steps(db, session.workflow_id)
    step = next((item for item in steps if item.id == session.step_id), None)

    if step is None:
        # The workflow was re-seeded under the session's feet. The run still happened; only our
        # claim about where it happened has gone stale.
        logger.info(
            "results: session step is no longer in the registry (workflow=%s step=%s)",
            session.workflow_id,
            session.step_id,
        )

        return None

    return StepOut(
        id=step.id,
        index=step.index,
        total=len(steps),
        is_final=step.is_final,
    )


async def _existing_report(db: AsyncSession, plan_id: str) -> ResultsResponse | None:
    """The answer this plan was already given, or None if it has not been reported."""
    statement = select(PlanReport.response).where(PlanReport.plan_id == plan_id)
    stored = (await db.execute(statement)).scalar_one_or_none()

    if stored is None:
        return None

    return ResultsResponse.model_validate(stored)


async def _claim_plan(
    db: AsyncSession,
    *,
    plan_id: str,
    session_id: uuid.UUID,
    response: ResultsResponse,
) -> bool:
    """
    Record this plan as reported, and say whether we were the ones who did it.

    `ON CONFLICT DO NOTHING` plus `RETURNING`, so the claim and the test are one statement: two
    requests racing on the same plan both reach here, and exactly one gets a row back. The loser
    writes nothing and replays the winner's answer, which is what makes "writes once" true under
    concurrency rather than only under polite clients.

    The race is settled by the database, not by timing: the second transaction's insert blocks on
    the first one's uncommitted row, then does nothing if that commit landed, or takes the claim
    if it rolled back.
    """
    statement = (
        insert(PlanReport)
        .values(
            plan_id=plan_id,
            session_id=session_id,
            response=response.model_dump(mode="json"),
        )
        .on_conflict_do_nothing(index_elements=[PlanReport.plan_id])
        .returning(PlanReport.plan_id)
    )

    return (await db.execute(statement)).scalar_one_or_none() is not None


async def _bump_session_counters(
    db: AsyncSession,
    session_id: uuid.UUID,
    totals: RunTotals,
) -> None:
    """
    Add this run's outcome to the session's running tally.

    One UPDATE with `column + value` rather than a read, an add and a write: two runs reported at
    once would otherwise each read the same number and one increment would vanish. The
    `set_updated_at()` trigger fires, which is honest — the session really was just touched.
    """
    columns = {
        f"results_{status}": getattr(Session, f"results_{status}") + getattr(totals, status)
        for status in RunTotals.model_fields
        if getattr(totals, status)
    }

    if not columns:
        return

    await db.execute(update(Session).where(Session.id == session_id).values(**columns))


async def _bump_counters(
    db: AsyncSession,
    *,
    payload: ResultsRequest,
    session: Session,
    totals: RunTotals,
) -> None:
    """
    Increment the durable counters — the only thing this endpoint writes that outlives the session.

    One statement for every metric the run touched, upserting on the whole key so a second run on
    the same step on the same day adds to the row rather than making another. `excluded.count` is
    this statement's own delta, which is what lets one INSERT both create a row at `n` and add `n`
    to a row that already exists.
    """
    metrics = results_utils.counter_metrics(payload, totals)

    if not metrics:
        return

    day = results_utils.counter_day()
    rows = [
        {
            "workflow_id": results_utils.scope_part(session.workflow_id),
            "step_id": results_utils.scope_part(session.step_id),
            "metric": metric,
            "day": day,
            "count": delta,
        }
        for metric, delta in metrics.items()
    ]

    statement = insert(ResultsCounter).values(rows)
    await db.execute(
        statement.on_conflict_do_update(
            index_elements=[
                ResultsCounter.workflow_id,
                ResultsCounter.step_id,
                ResultsCounter.metric,
                ResultsCounter.day,
            ],
            set_={"count": ResultsCounter.count + statement.excluded.count},
        ),
    )


async def _record_checkpoint(
    db: AsyncSession,
    *,
    payload: ResultsRequest,
    session_id: uuid.UUID,
) -> None:
    """
    Keep the checkpoint that stopped this run, if one did.

    The event holds the kind as reported; the counter holds it normalised. Both, because the event
    is the per-run detail someone reads while an application is live, and the counter is the
    evidence the safety layer ran at all — which has to survive the session to be worth anything.
    """
    if payload.checkpoint is None:
        return

    db.add(
        CheckpointEvent(
            session_id=session_id,
            plan_id=payload.plan_id,
            reason=payload.checkpoint.reason,
            after_index=payload.checkpoint.after_index,
        ),
    )


def _approved_by_id(actions: Sequence[PlannedActionOut]) -> dict[str, PlannedActionOut]:
    """The plan's approved actions, by the id the panel reported them under."""
    return {action.action_id: action for action in actions}


async def record_results(
    db: AsyncSession,
    user: AuthedUser,
    payload: ResultsRequest,
) -> ResultsResponse:
    """
    Record one run, once, and say what the user should do next.

    Raises nothing for a missing session, a missing plan, an expired plan or a plan already
    reported: every one of those is a `200` with `acknowledged: false`. The spec's error table is
    deliberate about it — this endpoint reports history, and bookkeeping we have lost is not the
    user's problem to see an error about.
    """
    session = await sessions_service.get_user_session(db, payload.session_id, user.id)

    if session is None:
        # Unknown, expired, or another user's — indistinguishable from here, and none of the three
        # is worth an error on a call that only writes history.
        logger.info("results: no live session for this user; nothing recorded")

        return _unacknowledged()

    replay = await _existing_report(db, payload.plan_id)

    if replay is not None:
        # Reported already. The stored answer, not a fresh one: a hint recomputed now could
        # disagree with the one the panel was given, against a session that has since moved on.
        return replay

    plan = plan_store.get_pending_plan(payload.plan_id, user.id)

    if plan is None or plan.session_id != session.id:
        # Expired, unknown, someone else's, or reported against the wrong session. The
        # in-process store is swept on a restart, so this is also what a report that outlived a
        # reload looks like — a gap in the numbers, never a wrong number.
        logger.info("results: no pending plan for this report; nothing recorded")

        return _unacknowledged(await _step_of(db, session))

    step = await _step_of(db, session)
    totals = results_utils.count_statuses(payload.results)
    rows = results_utils.action_log_rows(payload.results, _approved_by_id(plan.actions))

    response = ResultsResponse(
        acknowledged=True,
        recorded=len(rows),
        next_hint=results_utils.next_hint(
            results_utils.hint_code(
                totals,
                checkpoint=payload.checkpoint is not None,
                is_final=bool(step and step.is_final),
            ),
        ),
        step=step,
        run=totals,
    )

    claimed = await _claim_plan(
        db,
        plan_id=payload.plan_id,
        session_id=session.id,
        response=response,
    )

    if not claimed:
        # Another request claimed it between our read and our insert. Its answer is the one the
        # panel should see, and we write nothing.
        logger.info("results: plan claimed by a concurrent report; replaying it")

        return await _existing_report(db, payload.plan_id) or _unacknowledged(step)

    await sessions_service.record_actions(db, session.id, rows)
    await _record_checkpoint(db, payload=payload, session_id=session.id)
    await _bump_session_counters(db, session.id, totals)
    await _bump_counters(db, payload=payload, session=session, totals=totals)

    return response
