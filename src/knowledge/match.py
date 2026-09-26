"""
Where is the user: which workflow, and which step of it.

Pure functions. The caller passes the registry rows in; nothing here opens a connection,
makes a request, reads a clock, or calls a model. Same inputs, same outputs, forever —
which is what lets a hundred page variations be checked in a second instead of by clicking
through a portal.

This module must never import from `src.database`, `src.ai`, or anything that speaks HTTP.
A query acquired here would be paid on every page load and would make the judgement in this
file untestable without a database.
"""

from collections.abc import Sequence

from src.knowledge.constants import (
    FIRST_LABELS_BONUS,
    FIRST_LABELS_COUNT,
    MAX_URL_LENGTH,
    MIN_STEP_MARGIN,
    MIN_STEP_SCORE,
    SCORE_PRECISION,
)
from src.knowledge.normalize import normalize_label
from src.knowledge.schemas import StepCandidate, StepMatch, WorkflowMatch
from src.workflows.schemas import ActiveWorkflow, Step


def _without_fragment(url: str) -> str:
    """
    The URL up to its `#`.

    The fragment never reaches the server and a single-page portal rewrites it as the user
    scrolls, so matching on it would make the same page match differently twice. The query
    string stays: some portals carry the step number there.
    """
    return url.split("#", maxsplit=1)[0]


def match_workflow(
    url: str,
    workflows: Sequence[ActiveWorkflow],
) -> WorkflowMatch | None:
    """
    The workflow whose patterns accept this URL, or None.

    First match wins, in registry order — two workflows claiming one URL is a seed problem
    to fix in the data, not something to arbitrate on every request.

    Patterns are applied with `fullmatch`, so a pattern is anchored at both ends however it
    was written. `^https://portal\\.example\\.gov\\.ng` therefore does not accept
    `https://portal.example.gov.ng.evil.com/`, which a start-anchored match would.

    None is what B9 turns into `supported: false`.
    """
    if not url:
        return None

    # Refused, not truncated: a shortened URL can match a pattern the full one would not.
    if len(url) > MAX_URL_LENGTH:
        return None

    target = _without_fragment(url)

    for workflow in workflows:
        for index, pattern in enumerate(workflow.url_patterns):
            if pattern.fullmatch(target):
                return WorkflowMatch(workflow_id=workflow.id, pattern_index=index)

    return None


def _score_step(step: Step, present: frozenset[str]) -> tuple[float, list[str], list[str]]:
    """
    One step's score against the page, with its matched and missing labels.

    Coverage, not similarity: how much of what we know about this step is on the page. A
    page carrying extra fields we have never seen still matches its step, which is what
    keeps a portal's optional questions from breaking the match.

    Labels are returned in their stored form, because they are shown to a person.
    """
    matched: list[str] = []
    missing: list[str] = []

    for label in step.field_labels:
        target = matched if normalize_label(label) in present else missing
        target.append(label)

    known_count = len(step.field_labels)

    # A step with no labels tells us nothing about the page, so it scores nothing. Without
    # this the coverage below divides by zero.
    if known_count == 0:
        return 0.0, matched, missing

    coverage = len(matched) / known_count
    bonus = FIRST_LABELS_BONUS if _opening_labels_present(step, present) else 0.0
    score = round(min(1.0, coverage + bonus), SCORE_PRECISION)

    return score, matched, missing


def _opening_labels_present(step: Step, present: frozenset[str]) -> bool:
    """Whether the step's first two labels are both on the page."""
    opening = step.field_labels[:FIRST_LABELS_COUNT]

    if len(opening) < FIRST_LABELS_COUNT:
        return False

    return all(normalize_label(label) in present for label in opening)


def match_step(labels: Sequence[str], steps: Sequence[Step]) -> StepMatch:
    """
    The step of a workflow this page is showing.

    Always returns the best guess. `confident` is the part that matters: it is true only
    when the best step clears `MIN_STEP_SCORE` *and* beats the runner-up by
    `MIN_STEP_MARGIN`. Two steps at 0.62 and 0.60 is a coin toss, and B9 marks that
    low-confidence rather than acting on it.

    Page labels are compared as a set, so a portal repeating a label does not weight it.
    """
    if not steps:
        return StepMatch(
            step_id=None,
            score=0.0,
            confident=False,
            reason="no steps to match against",
        )

    present = frozenset(normalize_label(label) for label in labels if label.strip())

    # Sorted by score, then by the portal's own order — so two steps that tie resolve to
    # the earlier one rather than to whichever the database happened to return first.
    scored = sorted(
        ((step, *_score_step(step, present)) for step in steps),
        key=lambda row: (-row[1], row[0].index),
    )

    best_step, best_score, matched, missing = scored[0]

    # A single-step workflow has no runner-up, so the margin is measured against zero: the
    # only way to fail is on the score itself.
    runner_up = (
        StepCandidate(step_id=scored[1][0].id, score=scored[1][1])
        if len(scored) > 1
        else None
    )
    runner_up_score = runner_up.score if runner_up else 0.0
    margin = round(best_score - runner_up_score, SCORE_PRECISION)

    confident = best_score >= MIN_STEP_SCORE and margin >= MIN_STEP_MARGIN

    return StepMatch(
        step_id=best_step.id,
        score=best_score,
        confident=confident,
        matched_labels=tuple(matched),
        missing_labels=tuple(missing),
        runner_up=runner_up,
        reason=_reason(best_score, margin, confident=confident),
    )


def _reason(score: float, margin: float, *, confident: bool) -> str:
    """One line naming which condition decided the outcome, for a log."""
    if confident:
        return f"score {score} over {MIN_STEP_SCORE}, margin {margin} over {MIN_STEP_MARGIN}"

    if score < MIN_STEP_SCORE:
        return f"score {score} below {MIN_STEP_SCORE}"

    return f"margin {margin} below {MIN_STEP_MARGIN}"
