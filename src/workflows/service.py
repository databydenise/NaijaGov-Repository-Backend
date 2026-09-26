"""Workflow and step lookups."""

import time
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.workflows.constants import SUPPORTED_HOSTS_TTL_SECONDS
from src.workflows.models import Workflow, WorkflowStep
from src.workflows.utils import host_from_pattern

# (expires at, on the monotonic clock; hosts). Per process, which is fine: a stale entry
# lives at most one TTL, and the only thing that changes the answer is a seed run.
_supported_hosts_cache: tuple[float, list[str]] | None = None


async def list_active_workflows(db: AsyncSession) -> Sequence[Workflow]:
    """Every active workflow. B8 matches a page URL against their `url_patterns`."""
    statement = (
        select(Workflow).where(Workflow.is_active.is_(True)).order_by(Workflow.id)
    )

    result = await db.execute(statement)

    return result.scalars().all()


async def _load_registry_hosts(db: AsyncSession) -> list[str]:
    """Hosts named by active workflows' `url_patterns`, deduplicated and sorted."""
    statement = select(Workflow.url_patterns).where(Workflow.is_active.is_(True))
    result = await db.execute(statement)

    hosts = {
        host
        for patterns in result.scalars()
        for pattern in patterns
        if (host := host_from_pattern(pattern)) is not None
    }

    return sorted(hosts)


async def get_supported_hosts(db: AsyncSession) -> list[str]:
    """
    Hosts the extension can offer help on, for `/me`.

    Cached in memory for `SUPPORTED_HOSTS_TTL_SECONDS`. Dev hosts from config are added on
    every call rather than cached, and are empty in production.
    """
    global _supported_hosts_cache  # noqa: PLW0603  # a module-level cache is the point

    now = time.monotonic()

    if _supported_hosts_cache is None or _supported_hosts_cache[0] <= now:
        hosts = await _load_registry_hosts(db)
        _supported_hosts_cache = (now + SUPPORTED_HOSTS_TTL_SECONDS, hosts)

    registry_hosts = _supported_hosts_cache[1]
    extra = [host for host in settings.extra_supported_hosts if host not in registry_hosts]

    return [*registry_hosts, *extra]


async def get_steps(db: AsyncSession, workflow_id: str) -> Sequence[WorkflowStep]:
    """A workflow's steps in the order the portal presents them."""
    statement = (
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow_id)
        .order_by(WorkflowStep.index)
    )

    result = await db.execute(statement)

    return result.scalars().all()
