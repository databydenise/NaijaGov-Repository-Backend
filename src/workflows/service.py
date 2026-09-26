"""Workflow and step lookups."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.workflows.models import Workflow, WorkflowStep


async def list_active_workflows(db: AsyncSession) -> Sequence[Workflow]:
    """Every active workflow. B8 matches a page URL against their `url_patterns`."""
    statement = (
        select(Workflow).where(Workflow.is_active.is_(True)).order_by(Workflow.id)
    )

    result = await db.execute(statement)

    return result.scalars().all()


async def get_steps(db: AsyncSession, workflow_id: str) -> Sequence[WorkflowStep]:
    """A workflow's steps in the order the portal presents them."""
    statement = (
        select(WorkflowStep)
        .where(WorkflowStep.workflow_id == workflow_id)
        .order_by(WorkflowStep.index)
    )

    result = await db.execute(statement)

    return result.scalars().all()
