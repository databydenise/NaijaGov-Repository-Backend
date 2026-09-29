"""
Identifying a page: which workflow, which step, which rules.

The judgement lives in `knowledge/match.py` and the data in the registry cache. This layer
orchestrates and persists. On a warm cache it does one read and one write, both on
`sessions`; everything else is a pure function over data already in memory.
"""

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.context.constants import CACHE_WINDOW_SECONDS, UNSUPPORTED_MESSAGE
from src.context.schemas import (
    Checkpoint,
    ContextRequest,
    ContextResponse,
    Coverage,
    FieldHint,
    StepOut,
    WorkflowOut,
)
from src.context.utils import compute_page_hash
from src.knowledge import service as knowledge_service
from src.knowledge.match import match_step, match_workflow
from src.knowledge.normalize import normalize_label
from src.knowledge.schemas import RuleOut, StepMatch
from src.sessions import service as sessions_service
from src.sessions.models import Session
from src.workflows import service as workflows_service
from src.workflows.schemas import ActiveWorkflow, Step

logger = logging.getLogger(__name__)


def _is_fresh(session: Session, page_hash: str) -> bool:
    """
    Whether this session already answers for this exact page, recently enough.

    `updated_at` is maintained by a database trigger on every UPDATE, so it is the time of
    the last write and not of the last read. That is what makes the window meaningful: a
    cache hit performs no write, so a page held open for an hour stops being fresh and is
    re-matched, rather than the window sliding forward forever.
    """
    if session.page_hash != page_hash:
        return False

    updated_at = session.updated_at

    # The column is `timestamptz`, so asyncpg returns an aware value. Treated as UTC if it
    # ever arrives naive: subtracting a naive from an aware datetime raises, and a stale
    # cache check is not worth a 500 on the page-read path.
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    age = (datetime.now(UTC) - updated_at).total_seconds()

    # A negative age means the row is stamped in the future — clock skew between the
    # database and this process. Treated as not fresh, so the answer is recomputed rather
    # than served from a row we cannot place in time.
    return 0 <= age < CACHE_WINDOW_SECONDS


def _field_hints(payload: ContextRequest, rules: list[RuleOut]) -> list[FieldHint]:
    """
    Which fields on this page have a rule.

    Matched on the normalised label, the same comparison the step matcher makes, so a
    field the step matched is a field that can find its rule. A field with no rule is
    simply absent: "no rule, no claim" is a property of the data, and inventing a hint for
    an unmatched field is exactly the failure this project exists to avoid.
    """
    by_label = {
        normalize_label(rule.field_label): rule
        for rule in rules
        if rule.field_label is not None
    }

    hints = []

    for field in payload.fields:
        rule = by_label.get(normalize_label(field.label))

        if rule is None:
            continue

        hints.append(
            FieldHint(
                field_id=field.field_id,
                rule_id=rule.id,
                is_placeholder=rule.is_placeholder,
            ),
        )

    return hints


def _checkpoint(payload: ContextRequest) -> Checkpoint:
    """
    That a checkpoint exists, and how many fields it covers.

    The count comes from the content script's flags plus any field it marked sensitive,
    because the two are reported separately and a page can use either. Which fields, and
    why, stay out of the response and out of the log.

    A flag naming a **navigation link** is deliberately not counted. A portal's masthead
    carries "Make a payment" on every page of the site, including the pages that have no
    payment on them, so counting it would put the panel into a checkpoint on a landing page
    and stop the Copilot before it has done anything. The link is still flagged, and the
    model is still shown it as BLOCKED — that is field-level sensitivity, the same call the
    extension makes. Only a control on this page gates this page.
    """
    link_ids = {link.field_id for link in payload.links}

    flagged = {
        flag.field_id for flag in payload.sensitive_flags if flag.field_id not in link_ids
    }
    flagged.update(field.field_id for field in payload.fields if field.sensitive)

    return Checkpoint(present=bool(flagged), count=len(flagged))


async def _unsupported(
    db: AsyncSession,
    user_id: uuid.UUID,
    payload: ContextRequest,
    page_hash: str,
) -> ContextResponse:
    """
    A page outside the registry.

    A `200`, not an error: the panel offers explain-only help, and a page that is simply
    not seeded yet should not put it into an error state. The session is still written, so
    the tab has one and chat can proceed.
    """
    session = await sessions_service.upsert_tab_session(
        db,
        user_id=user_id,
        tab_id=payload.tab_id,
        workflow_id=None,
        step_id=None,
        page_hash=page_hash,
        cached_rule_ids=[],
    )

    return ContextResponse(
        session_id=session.id,
        supported=False,
        page_hash=page_hash,
        message=UNSUPPORTED_MESSAGE,
    )


def _page_labels(payload: ContextRequest) -> list[str]:
    """
    What the matcher compares against: the field labels, and the headings.

    Headings are included because a step's name often appears as one, and a page rendered
    down to two visible fields still carries its heading.
    """
    return [field.label for field in payload.fields if field.label] + payload.headings


async def read_context(
    db: AsyncSession,
    user_id: uuid.UUID,
    payload: ContextRequest,
) -> ContextResponse:
    """
    Identify the page, load its rules, and save the session.

    Returns the same answer for the same page whether or not it came from the cache: the
    cache skips the database write, not the matching, which is pure and costs microseconds.
    """
    page_hash = compute_page_hash(payload.url, payload.fields)

    existing = await sessions_service.get_session_for_tab(db, user_id, payload.tab_id)
    cached = existing is not None and _is_fresh(existing, page_hash)

    workflows = await workflows_service.list_active_workflows(db)
    workflow_match = match_workflow(payload.url, workflows)

    if workflow_match is None:
        if cached and existing is not None:
            return ContextResponse(
                session_id=existing.id,
                supported=False,
                page_hash=page_hash,
                cached=True,
                message=UNSUPPORTED_MESSAGE,
            )

        return await _unsupported(db, user_id, payload, page_hash)

    workflow = next(item for item in workflows if item.id == workflow_match.workflow_id)
    steps = await workflows_service.get_steps(db, workflow.id)
    step_match = match_step(_page_labels(payload), steps)

    if step_match.step_id is None:
        # A seeded workflow with no steps. Nothing to be confident about, and no rules to
        # load, but the page is still a supported one.
        return await _unsupported(db, user_id, payload, page_hash)

    step = next(item for item in steps if item.id == step_match.step_id)
    rules = await knowledge_service.get_cached_rules_for_step(db, workflow.id, step.id)

    session = (
        existing
        if cached and existing is not None
        else await sessions_service.upsert_tab_session(
            db,
            user_id=user_id,
            tab_id=payload.tab_id,
            workflow_id=workflow.id,
            step_id=step.id,
            page_hash=page_hash,
            cached_rule_ids=[rule.id for rule in rules],
        )
    )

    return _build_response(
        session_id=session.id,
        page_hash=page_hash,
        cached=cached,
        workflow=workflow,
        step=step,
        total_steps=len(steps),
        step_match=step_match,
        payload=payload,
        rules=rules,
    )


def _build_response(  # noqa: PLR0913  # every argument is a distinct part of the answer
    *,
    session_id: uuid.UUID,
    page_hash: str,
    cached: bool,
    workflow: ActiveWorkflow,
    step: Step,
    total_steps: int,
    step_match: StepMatch,
    payload: ContextRequest,
    rules: list[RuleOut],
) -> ContextResponse:
    """Assemble the response. Pure: everything it needs has already been read."""
    return ContextResponse(
        session_id=session_id,
        supported=True,
        page_hash=page_hash,
        cached=cached,
        workflow=WorkflowOut(id=workflow.id, name=workflow.name, agency=workflow.agency),
        step=StepOut(
            id=step.id,
            name=step.name,
            index=step.index,
            total=total_steps,
            is_final=step.is_final,
        ),
        confidence="high" if step_match.confident else "low",
        coverage=Coverage(
            matched=len(step_match.matched_labels),
            missing=list(step_match.missing_labels),
        ),
        field_hints=_field_hints(payload, rules),
        checkpoint=_checkpoint(payload),
        rules_loaded=len(rules),
    )
