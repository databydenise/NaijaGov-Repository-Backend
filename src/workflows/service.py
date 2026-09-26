"""
Workflow and step lookups.

The registry changes only when the seed runs, so the two reads on the hot path are cached
for five minutes and their regexes are compiled once. A stale copy costs at most one TTL
after a re-seed; recompiling every pattern on every page load costs `/context` part of its
budget on every request.
"""

import logging
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache import cached
from src.config import settings
from src.workflows.constants import (
    ACTIVE_WORKFLOWS_CACHE_KEY,
    REGISTRY_CACHE_TTL_SECONDS,
    STEPS_CACHE_KEY_PREFIX,
)
from src.workflows.models import Workflow, WorkflowStep
from src.workflows.schemas import ActiveWorkflow, Step
from src.workflows.utils import host_from_pattern, is_local_host

logger = logging.getLogger(__name__)


async def _load_active_workflows(db: AsyncSession) -> list[ActiveWorkflow]:
    """Read the active workflows and compile their patterns."""
    statement = (
        select(Workflow).where(Workflow.is_active.is_(True)).order_by(Workflow.id)
    )
    result = await db.execute(statement)

    return [_to_active_workflow(workflow) for workflow in result.scalars()]


def _to_active_workflow(workflow: Workflow) -> ActiveWorkflow:
    """
    One row, with its patterns compiled.

    A pattern that does not compile is dropped with a warning rather than raised: the seed
    already rejects those, so reaching here means the row was written another way, and one
    bad pattern should not take `/context` down for every other workflow.
    """
    patterns = []

    for pattern in workflow.url_patterns:
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            logger.warning(
                "Workflow %s has a url_pattern that does not compile; skipping it",
                workflow.id,
            )

    return ActiveWorkflow(
        id=workflow.id,
        agency=workflow.agency,
        name=workflow.name,
        url_patterns=tuple(patterns),
    )


async def list_active_workflows(db: AsyncSession) -> list[ActiveWorkflow]:
    """Every active workflow, patterns compiled. Cached for five minutes."""
    return await cached(
        ACTIVE_WORKFLOWS_CACHE_KEY,
        REGISTRY_CACHE_TTL_SECONDS,
        lambda: _load_active_workflows(db),
    )


async def _load_steps(db: AsyncSession, workflow_id: str) -> list[Step]:
    """Read one workflow's steps in portal order."""
    statement = (
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow_id)
        .order_by(WorkflowStep.index)
    )
    result = await db.execute(statement)

    return [
        Step(
            id=step.id,
            workflow_id=step.workflow_id,
            name=step.name,
            index=step.index,
            field_labels=tuple(step.field_labels),
            is_final=step.is_final,
        )
        for step in result.scalars()
    ]


async def get_steps(db: AsyncSession, workflow_id: str) -> list[Step]:
    """A workflow's steps in the order the portal presents them. Cached for five minutes."""
    return await cached(
        f"{STEPS_CACHE_KEY_PREFIX}{workflow_id}",
        REGISTRY_CACHE_TTL_SECONDS,
        lambda: _load_steps(db, workflow_id),
    )


async def get_supported_hosts(db: AsyncSession) -> list[str]:
    """
    Hosts the extension can offer help on, for `/me`.

    Read from the same cached registry the matcher uses, so the panel and the matcher can
    never disagree about which portals are supported. Local hosts are dropped in production:
    a seeded `localhost` pattern is there for development, and advertising it to a real
    user would be nonsense.
    """
    workflows = await list_active_workflows(db)

    hosts = {
        host
        for workflow in workflows
        for pattern in workflow.url_patterns
        if (host := host_from_pattern(pattern.pattern)) is not None
    }

    if settings.env == "production":
        hosts = {host for host in hosts if not is_local_host(host)}

    extra = [host for host in settings.extra_supported_hosts if host not in hosts]

    return [*sorted(hosts), *extra]
