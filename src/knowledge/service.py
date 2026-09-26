"""Rule retrieval.

A lookup by workflow, step, and field label — not similarity search. At this size a lookup
is more accurate, and it is explainable when a judge asks why a given rule came back.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.models import Rule


async def get_rules_for_step(
    db: AsyncSession,
    workflow_id: str,
    step_id: str,
) -> Sequence[Rule]:
    """Rules for one step, plus the workflow-wide rules that apply on every step.

    Workflow-wide rules (`step_id IS NULL`) are included because a requirement that holds
    across the whole application still holds on this step, and leaving them out would make
    the answer say "no rule" when there is one.

    An empty result is a real answer: no rule, no claim.
    """
    statement = (
        select(Rule)
        .where(
            Rule.workflow_id == workflow_id,
            (Rule.step_id == step_id) | Rule.step_id.is_(None),
        )
        .order_by(Rule.id)
    )

    result = await db.execute(statement)

    return result.scalars().all()
