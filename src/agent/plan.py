"""
The plan turn: a rendered context in, a guarded plan or a reason out.

This is the sequence the spec describes, in order: take the session's turn slot, check the day's
quota, run the calls (`turn.py`), guard the answer (P3), repair once if the diagnosis earns it,
release, measure, return.

Three properties hold throughout, and each one is load-bearing:

- **Nothing is raised at the caller.** Every failure is a `TurnFailure` with a code and a sentence
  the panel prints verbatim. A caller that has to catch something eventually will not.
- **Nothing is persisted.** No session row, no action log, no extracted data. P5 decides what to
  store, which is what lets every check in this task run with no database and no network.
- **Nothing ungrounded survives.** If a claim cannot be verified and the repair cannot run — out
  of budget, or already used on a schema failure — the final guard pass is still made with
  `repair_attempted=True`, so the claim is replaced rather than shown. There is no path on which
  an unsupported fee reaches a citizen because the clock ran out.
"""

import logging
from collections.abc import Collection, Mapping, Sequence

from src.agent.calls import call_model, has_budget
from src.agent.exceptions import TurnAborted
from src.agent.locks import turn_lock
from src.agent.quota import check_quota, record_usage
from src.agent.repair import RepairReason, repair_instruction, repair_messages
from src.agent.schemas import PlanOutcome, PlanTurn
from src.agent.service import (
    PLAN_SCHEMA_NAME,
    as_failure,
    new_state,
    turn_deadline_from,
)
from src.agent.telemetry import TurnState, build_telemetry, log_turn, phase
from src.agent.tools import RetrieveCallable, ToolRun
from src.agent.turn import converse, parse_reply
from src.ai.client import Message, ModelClient
from src.ai.context import PlanContext
from src.ai.prompt_loader import SYSTEM_PLAN
from src.ai.schemas import PLAN_RESPONSE_SCHEMA, PlanResponse
from src.context.schemas import PageButton, PageField
from src.guard.schemas import GuardedPlan
from src.guard.service import guard_plan

logger = logging.getLogger(__name__)


def _plan_repair_reason(plan: GuardedPlan, response: PlanResponse) -> RepairReason | None:
    """
    Whether this plan earns the one repair, and for which diagnosis.

    Only two things here do. An `unverified` verdict, which the guard itself asks for. And a plan
    whose every action was thrown out as *malformed* — structurally unusable, so the model can fix
    it. A plan rejected for legitimate reasons (a sensitive field, an id that is not on the page)
    earns nothing: the guard was right, and asking again invites an argument with it.
    """
    if plan.repair_requested:
        return "unverified"

    every_action_malformed = plan.rejected and all(
        rejected.code == "MALFORMED" for rejected in plan.rejected
    )

    if response.actions and not plan.approved and every_action_malformed:
        return "malformed_actions"

    return None


async def run_plan_turn(
    *,
    context: PlanContext,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton] = (),
    profile: Mapping[str, str | None],
    chat_values: Mapping[str, str] | None = None,
    blocked_field_ids: Collection[str] = (),
    retrieve: RetrieveCallable,
    client: ModelClient,
    session_id: str,
    user_id: str,
    deadline: float | None = None,
) -> PlanOutcome:
    """
    Run one plan turn: the model, retrieval, the guard, and at most one repair.

    `context` is P2's rendering — the page block, the user-data block, the history and the message,
    already under its token budget. `fields`, `buttons` and `blocked_field_ids` are *this* turn's
    snapshot, the one the request was made against; the guard checks the answer against them and
    nothing else. `profile` and `chat_values` hold the real values, which the model never sees and
    the guard substitutes after validation.

    `retrieve` and `client` are injected — `search_government_information` and `OpenAIClient` in
    production, a stub and a scripted fake in every check.
    """
    state = new_state("plan", client, session_id, user_id)
    state.context_tokens = context.estimated_tokens
    state.truncated = tuple(context.truncated)
    turn_deadline = turn_deadline_from(deadline)
    run = ToolRun()

    try:
        async with turn_lock(session_id):
            check_quota(user_id)

            response, messages = await converse(
                model=PlanResponse,
                schema=PLAN_RESPONSE_SCHEMA,
                schema_name=PLAN_SCHEMA_NAME,
                system_prompt=SYSTEM_PLAN,
                context_text=context.text,
                client=client,
                retrieve=retrieve,
                run=run,
                state=state,
                deadline=turn_deadline,
            )

            def check(subject: PlanResponse, *, second_pass: bool) -> GuardedPlan:
                with phase(state, "guard"):
                    return guard_plan(
                        subject,
                        fields=fields,
                        buttons=buttons,
                        profile=profile,
                        chat_values=chat_values,
                        chunks=run.chunks,
                        blocked_field_ids=blocked_field_ids,
                        repair_attempted=second_pass or state.repair_ran,
                    )

            plan = check(response, second_pass=False)
            reason = _plan_repair_reason(plan, response)

            if reason is not None:
                subject = response

                if not state.repair_ran and has_budget(turn_deadline):
                    subject = await _repair_plan(
                        response,
                        reason=reason,
                        codes=plan.report.rejection_codes,
                        messages=messages,
                        client=client,
                        state=state,
                        deadline=turn_deadline,
                    )

                # Whether or not the repair ran, the second pass is final: an unverified claim is
                # replaced here rather than shown to the user because there was no time to ask.
                plan = check(subject, second_pass=True)

            record = build_telemetry(
                state,
                grounding=plan.grounding,
                approved=plan.report.approved,
                rejected=plan.report.rejected,
            )
            log_turn(record)

            return PlanTurn(
                plan=plan,
                grounding=plan.grounding,
                chunks=tuple(run.chunks),
                telemetry=record,
            )
    except TurnAborted as abort:
        return as_failure(abort, state)
    finally:
        # Recorded for a failed turn too: a call that timed out after the provider had read the
        # prompt still cost what it cost.
        record_usage(user_id, state.total_tokens)


async def _repair_plan(
    response: PlanResponse,
    *,
    reason: RepairReason,
    codes: Mapping[str, int],
    messages: Sequence[Message],
    client: ModelClient,
    state: TurnState,
    deadline: float,
) -> PlanResponse:
    """
    The one repair attempt for a plan. Returns the repaired response, or the original.

    A model failure here does not fail the turn. The first answer has already been guarded and its
    actions are sound; losing the whole plan because the provider went down between two calls
    would be the wrong trade for the person watching the panel.
    """
    state.repair_ran = True
    state.repair_reason = reason

    try:
        reply = await call_model(
            client=client,
            state=state,
            messages=repair_messages(
                messages,
                repair_instruction(reason, rejection_codes=codes),
            ),
            deadline=deadline,
            response_schema=PLAN_RESPONSE_SCHEMA,
            schema_name=PLAN_SCHEMA_NAME,
        )
    except TurnAborted as abort:
        logger.info("plan repair abandoned (%s); keeping the first answer", abort.code)

        return response

    repaired, _ = parse_reply(reply, PlanResponse, state)

    return repaired if repaired is not None else response
