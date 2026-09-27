"""
Where the question was asked: the session, the workflow, the step, and the corpus's vintage.

Split out of `service.py` because it is a different kind of work — every function here is a read
that may come back with nothing, and every "nothing" is a degradation the endpoint carries on
through rather than an error it raises. Keeping that reasoning in one place is what stops it being
re-decided, slightly differently, at each of the five points where a lookup can come up empty.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.context import StepView
from src.documents.service import latest_ingested_at
from src.explain.constants import UNKNOWN_CORPUS_STAMP, UNKNOWN_STEP_NAME
from src.explain.schemas import ExplainScope
from src.sessions import service as sessions_service
from src.workflows import service as workflows_service

logger = logging.getLogger(__name__)


async def resolve_scope(
    db: AsyncSession,
    session_id: uuid.UUID | None,
    user_id: uuid.UUID,
) -> ExplainScope:
    """
    Where the question was asked, as far as the registry knows.

    Every early return here is a degradation, not an error. No session id, an expired session,
    another account's session id, a page outside the registry, a step that has been re-seeded away
    — each produces a scope that claims nothing, which means the search is not narrowed to an
    agency and the prompt gets no step line. `/plan` refuses in most of these cases because it is
    about to fill a form; this endpoint is about to say a sentence about one field, and refusing to
    say it would make Explain the feature that stops working first.

    Both registry reads are served from the five-minute cache, so this is usually one query.
    """
    if session_id is None:
        return ExplainScope()

    session = await sessions_service.get_user_session(db, session_id, user_id)

    if session is None:
        # Unknown, expired, or someone else's — indistinguishable here on purpose, and none of
        # the three is worth failing a request that writes nothing.
        logger.info("explain: no live session for this user; answering without an agency filter")

        return ExplainScope()

    if session.workflow_id is None or session.step_id is None:
        return ExplainScope()

    workflows = await workflows_service.list_active_workflows(db)
    workflow = next((item for item in workflows if item.id == session.workflow_id), None)

    if workflow is None:
        return ExplainScope()

    steps = await workflows_service.get_steps(db, workflow.id)
    step = next((item for item in steps if item.id == session.step_id), None)

    if step is None:
        # The workflow was re-seeded under the session's feet. Its agency is still the right
        # filter; only our claim about where the user is has gone stale.
        return ExplainScope(
            workflow_id=workflow.id,
            agency=workflow.agency,
            service=workflow.name,
        )

    return ExplainScope(
        workflow_id=workflow.id,
        step_id=step.id,
        agency=workflow.agency,
        service=workflow.name,
        step=StepView(name=step.name, index=step.index, total=len(steps)),
    )


async def corpus_stamp(db: AsyncSession, scope: ExplainScope) -> str:
    """
    The vintage of the material this answer could be grounded in, for the cache key.

    This is the whole of cache invalidation. Re-ingesting an agency's material moves the timestamp,
    every key built from it changes, and the rows written against the old corpus become unreachable
    — no purge, no delete path, nothing to remember to run. A corpus with no ingest date, or one
    the database could not be asked about, gets a fixed marker instead, so an answer written during
    an outage cannot be served once the corpus is readable again.
    """
    ingested_at = await latest_ingested_at(db, scope.agency)

    return ingested_at.isoformat() if ingested_at else UNKNOWN_CORPUS_STAMP


def step_view(scope: ExplainScope) -> StepView | None:
    """The step line for the prompt. Absent rather than invented for a page with no workflow."""
    if scope.step is not None:
        return scope.step

    return StepView(name=UNKNOWN_STEP_NAME, index=1, total=1) if scope.workflow_id else None
