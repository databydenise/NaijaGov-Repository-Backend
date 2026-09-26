"""Rule retrieval.

A lookup by workflow, step, and field label — not similarity search. At this size a lookup
is more accurate, and it is explainable when a judge asks why a given rule came back.
"""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.models import Rule
from src.knowledge.schemas import RuleOut


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


async def get_rule_for_field(
    db: AsyncSession,
    workflow_id: str,
    step_id: str,
    field_label: str,
) -> Rule | None:
    """
    The rule covering one field, or None.

    Matched case-insensitively on the label but not normalised further: the seed holds the
    label exactly as the page renders it, and the caller passes what it read from the page.
    A field-scoped rule wins over a step-wide one, which is what `field_label IS NULL`
    sorting last does.

    None is a real answer, and the caller says "no rule" rather than filling the gap.
    """
    statement = (
        select(Rule)
        .where(
            Rule.workflow_id == workflow_id,
            (Rule.step_id == step_id) | Rule.step_id.is_(None),
            func.lower(Rule.field_label) == field_label.strip().lower(),
        )
        .order_by(Rule.step_id.is_(None), Rule.id)
        .limit(1)
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


def to_rule_out(rule: Rule) -> RuleOut:
    """A stored rule as the API shows it, provenance included."""
    return RuleOut(
        id=rule.id,
        topic=rule.topic,
        field_label=rule.field_label,
        requirement=rule.requirement,
        source_url=rule.source_url,
        last_checked=rule.last_checked,
        is_placeholder=rule.is_placeholder,
    )


async def count_placeholder_rules(db: AsyncSession) -> int:
    """How many stored rules are demo content. Reported by `/health`."""
    statement = select(func.count()).select_from(Rule).where(Rule.is_placeholder.is_(True))
    result = await db.execute(statement)

    return result.scalar_one()
