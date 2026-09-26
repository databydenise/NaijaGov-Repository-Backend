"""
How a rule leaves this service.

No endpoint returns one yet — `/explain` is B9 — but the shape is fixed here so that when
one does, it cannot be shipped without its provenance. Every field below is part of the
answer, not decoration: a requirement shown without its source, its date, and whether it
is demo content is exactly the failure this project is built to avoid.
"""

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel


class RuleOut(BaseModel):
    """One rule, with the provenance that must travel with it."""

    id: str
    topic: str
    field_label: str | None
    requirement: str

    # Shown with every answer that uses this rule. Never omitted to save space.
    source_url: str
    last_checked: date

    # True means invented demo content. The panel shows it in place of the source line, so
    # a plausible-looking requirement is never mistaken for official guidance.
    is_placeholder: bool


@dataclass(frozen=True)
class WorkflowMatch:
    """Which workflow a URL belongs to, and which of its patterns said so.

    `pattern_index` is here to be logged. When a page matches the wrong workflow, the
    question is always which pattern was too broad, and without this the answer takes a
    person reading regexes.
    """

    workflow_id: str
    pattern_index: int


@dataclass(frozen=True)
class StepCandidate:
    """One step's score. Returned as the runner-up, so a near-miss is visible in a log."""

    step_id: str
    score: float


@dataclass(frozen=True)
class StepMatch:
    """
    Which step of a workflow a page is showing, and how sure we are.

    Never a bare boolean: `score` is what B10 uses to decide whether to spend a model call,
    and `matched_labels` / `missing_labels` are what the panel means by "3 of 6 fields
    found". Both label lists hold the step's own stored labels, not their normalised forms,
    because they are shown to a person.

    `step_id` is None only when there were no steps to choose between.
    """

    step_id: str | None
    score: float
    confident: bool
    matched_labels: tuple[str, ...] = ()
    missing_labels: tuple[str, ...] = ()
    runner_up: StepCandidate | None = None

    # Why the answer is what it is, for a log line. Not shown to a user.
    reason: str = ""
