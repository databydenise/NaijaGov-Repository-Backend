"""
Pure helpers for `/results`: counting, the hint table, and the rows a report turns into.

No database, no clock beyond the caller's own date, no model, no randomness. Everything here is a
function of the request and the plan it reports on, which is what lets the whole response shape
and every counter update be checked without a database.

The one judgement in this module is `next_hint`, and it is a table rather than a paragraph of
`if`s so that the priority order the spec sets is the order you read.
"""

import re
from collections.abc import Mapping, Sequence
from datetime import date

from src.plan.schemas import PlannedActionOut
from src.results.constants import (
    ABORT_METRIC_PREFIX,
    CHECKPOINT_KIND_ALIASES,
    CHECKPOINT_KIND_OTHER,
    CHECKPOINT_KINDS,
    CHECKPOINT_METRIC_PREFIX,
    CHECKPOINT_PENDING,
    FINAL_REVIEW,
    FIX_FIRST,
    NEXT_HINTS,
    REVIEW_CONTINUE,
    UNFINISHED_STATUSES,
    UNKNOWN_SCOPE_PART,
)
from src.results.schemas import NextHintOut, ResultEntry, ResultsRequest, RunTotals

# Anything that is not a letter or a digit, for folding a checkpoint kind onto one spelling.
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def count_statuses(results: Sequence[ResultEntry]) -> RunTotals:
    """This run's outcome, counted by status."""
    counts = {status: 0 for status in RunTotals.model_fields}

    for entry in results:
        counts[entry.status] += 1

    return RunTotals(**counts)


def hint_code(totals: RunTotals, *, checkpoint: bool, is_final: bool) -> str:
    """
    Which hint this run earns, in the spec's own priority order.

    A checkpoint outranks everything because it is the only outcome where the user is already
    being asked for something specific. Unfinished fields outrank finality because telling someone
    to submit a form with two empty required fields is worse advice than telling them to fix them.
    """
    if checkpoint:
        return CHECKPOINT_PENDING

    if any(getattr(totals, status) for status in UNFINISHED_STATUSES):
        return FIX_FIRST

    if is_final:
        return FINAL_REVIEW

    return REVIEW_CONTINUE


def next_hint(code: str) -> NextHintOut:
    """The hint for a code, with the sentence the panel prints verbatim."""
    return NextHintOut(code=code, message=NEXT_HINTS[code])


def checkpoint_kind(raw: str) -> str:
    """
    A reported checkpoint kind, folded onto the one spelling the counters use.

    "One-Time Code", "one_time_code" and "OTP" are one kind and have to count as one row.
    Anything the list does not know becomes `other` rather than a row of its own — and rather
    than a rejected report, because a word is not worth losing a run's results over.
    """
    folded = _NOT_ALNUM.sub(" ", raw.casefold()).strip()

    if not folded:
        return CHECKPOINT_KIND_OTHER

    aliased = CHECKPOINT_KIND_ALIASES.get(folded, folded.replace(" ", "_"))

    return aliased if aliased in CHECKPOINT_KINDS else CHECKPOINT_KIND_OTHER


def action_log_rows(
    results: Sequence[ResultEntry],
    approved: Mapping[str, PlannedActionOut],
) -> list[tuple[str, str, str, str | None]]:
    """
    The `action_log` rows a report turns into: `(action_type, field_id, status, reason)`.

    The **plan** supplies the action type and the field id, not the request. The client's
    `field_id` is read for nothing, and that is the point: a row whose type or target came from
    the caller could disagree with what was actually approved, and every metric built on this
    table would inherit the disagreement. An `action_id` the plan does not contain is skipped —
    there is no honest `action_type` to write for it.
    """
    rows: list[tuple[str, str, str, str | None]] = []

    for entry in results:
        action = approved.get(entry.action_id)

        if action is None or not action.field_id:
            continue

        rows.append((action.type, action.field_id, entry.status, entry.reason))

    return rows


def counter_metrics(payload: ResultsRequest, totals: RunTotals) -> dict[str, int]:
    """
    Every durable counter this report increments, as `{metric: delta}`.

    One status per key with a non-zero count, plus a namespaced key for a checkpoint and one for
    an abort. Namespaced so a checkpoint kind can never collide with a status — `payment` the
    checkpoint and a hypothetical `payment` status would otherwise share a row and mean nothing.
    """
    metrics = {
        status: count
        for status in RunTotals.model_fields
        if (count := getattr(totals, status))
    }

    if payload.checkpoint is not None:
        kind = checkpoint_kind(payload.checkpoint.reason)
        metrics[f"{CHECKPOINT_METRIC_PREFIX}{kind}"] = 1

    if payload.aborted is not None:
        metrics[f"{ABORT_METRIC_PREFIX}{payload.aborted}"] = 1

    return metrics


def scope_part(value: str | None) -> str:
    """A workflow or step id for a counter key, or the marker for a page with neither."""
    return value or UNKNOWN_SCOPE_PART


def counter_day() -> date:
    """
    Today, in UTC.

    UTC so a day does not shift with a deployment's timezone, the same reason `agent/quota.py`
    keys its daily spend on a UTC date. A function rather than a default argument, because a
    default argument would freeze the date at import.
    """
    from datetime import UTC, datetime  # noqa: PLC0415  # kept local so the module stays pure

    return datetime.now(UTC).date()
