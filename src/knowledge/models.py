"""Curated rules.

`source_url` and `last_checked` are NOT NULL on purpose: a rule without a source is not a
rule, and the constraint is what makes "no rule, no claim" true in the data rather than
only in the prompt.
"""

from datetime import date

from sqlalchemy import Date, ForeignKey, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin


class Rule(TimestampMixin, Base):
    __tablename__ = "rules"
    __table_args__ = (
        # The two lookups B9 and /explain make.
        Index("ix_rules_workflow_step", "workflow_id", "step_id"),
        Index("ix_rules_workflow_field_label", "workflow_id", "field_label"),
    )

    # `cac_bn.proprietor.passport_photo`
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    workflow_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Null for a rule that applies to the whole workflow rather than one step. A step
    # that goes away takes its step-scoped rules with it.
    step_id: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("workflow_steps.id", ondelete="CASCADE"),
    )

    topic: Mapped[str] = mapped_column(Text, nullable=False)

    # Null for a rule about the step in general rather than one field.
    field_label: Mapped[str | None] = mapped_column(Text)

    requirement: Mapped[str] = mapped_column(Text, nullable=False)

    # Shown with every answer that uses this rule.
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    last_checked: Mapped[date] = mapped_column(Date, nullable=False)
