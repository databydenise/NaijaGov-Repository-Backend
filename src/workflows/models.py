"""Portals and their steps.

Ids are readable text (`cac_bn`, `cac_bn.proprietor`), not UUIDs: they appear in seeds,
prompts, logs, and bug reports, and `cac_bn.proprietor` in a log line is worth more than a
UUID. `url_patterns` holds regex strings matched in B8, which keeps adding a portal a seed
change rather than a deploy.
"""

from sqlalchemy import Boolean, ForeignKey, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base, TimestampMixin


class Workflow(TimestampMixin, Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    agency: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)

    url_patterns: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )


class WorkflowStep(TimestampMixin, Base):
    __tablename__ = "workflow_steps"
    __table_args__ = (
        UniqueConstraint("workflow_id", "index", name="uq_workflow_steps_workflow_index"),
    )

    # `<workflow_id>.<key>`
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    workflow_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)

    field_labels: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )

    # What lets /context warn that the next button submits the application.
    is_final: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
