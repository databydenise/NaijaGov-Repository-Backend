"""
Pure mapping from what the guard produced to what the panel renders.

No database, no clock, no model, no randomness — everything here is a function of the guarded
plan and the snapshot it was checked against, which is what lets the whole response shape be
checked without a turn. The snapshot is also read the other way here, for what the guard needs
from it: which fields are checkpoints.

The one judgement in this module is how provenance is worded, and it is a judgement about
trust rather than presentation: a row that says "Your profile" beside a value is a row the user
can check in a second, and one that says nothing is a value they have to take on faith.
"""

from collections.abc import Mapping, Sequence

from src.agent.schemas import PlanTurn
from src.ai.schemas import Citation
from src.context.schemas import PageButton, PageField
from src.documents.schemas import RetrievedChunk
from src.guard.schemas import ApprovedAction, GuardedPlan, RejectedAction
from src.plan.constants import SOURCE_LABELS, UNKNOWN_SOURCE_LABEL
from src.plan.schemas import (
    CitationOut,
    MissingItemOut,
    PlannedActionOut,
    PlanRequest,
    PlanResponse,
    RejectedActionOut,
    StepOut,
)


def labels_by_field_id(
    fields: Sequence[PageField],
    buttons: Sequence[PageButton],
) -> dict[str, str]:
    """
    Every id on the page and what it is called, buttons included.

    Buttons are here because a `clickSafe` or a blocked-button `pause` names one, and a preview
    row reading "I don't press this button myself" is clearer beside the button's own text.
    """
    labels = {page_field.field_id: page_field.label for page_field in fields}
    labels.update({button.field_id: button.text for button in buttons})

    return labels


def source_label(source: str | None) -> str | None:
    """
    `profile.email` → "Your profile". None for an action that writes nothing.

    An unrecognised prefix falls back to a sentence that claims nothing rather than showing our
    own key to the user. The guard only ever produces `profile.*` or `chat.*`, so the fallback is
    unreachable today and exists so a new source kind cannot leak an internal name into a panel.
    """
    if source is None:
        return None

    prefix, _, _ = source.partition(".")

    return SOURCE_LABELS.get(prefix, UNKNOWN_SOURCE_LABEL)


def to_action_out(
    index: int,
    action: ApprovedAction,
    labels: Mapping[str, str],
) -> PlannedActionOut:
    """
    One approved action as a preview row.

    `action_id` is positional — `a1`, `a2` — and means nothing beyond "the nth row of this plan".
    It exists so the panel and `/results` can refer to a row without repeating a field id, which
    is not unique across a plan: one field may be highlighted and filled.
    """
    label = labels.get(action.field_id) if action.field_id else None

    return PlannedActionOut(
        action_id=f"a{index}",
        type=action.type,
        field_id=action.field_id,
        label=label or None,
        value=action.value,
        checked=action.checked,
        source=source_label(action.source),
        source_ref=action.source,
        reason=action.reason,
        note=action.note,
        suspicious=action.suspicious,
    )


def to_rejected_out(
    rejected: RejectedAction,
    labels: Mapping[str, str],
) -> RejectedActionOut:
    """One refusal as the panel shows it: which field, and the guard's sentence. Never the code."""
    label = labels.get(rejected.field_id) if rejected.field_id else None

    return RejectedActionOut(
        field_id=rejected.field_id,
        label=label or None,
        reason=rejected.message,
    )


def to_citations_out(
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
) -> list[CitationOut]:
    """
    Verified citations, with the title and date of the chunk each one refers to.

    The guard has already dropped any citation whose `chunk_id` was not retrieved, so a lookup
    that misses here cannot happen from a guarded plan. It is still handled: the alternative is a
    `KeyError` on the response path, and a citation with a URL and no title is worth more to a
    person than a 500.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rendered: list[CitationOut] = []

    for citation in citations:
        chunk = by_id.get(citation.chunk_id)
        ingested_at = chunk.ingested_at if chunk else None

        rendered.append(
            CitationOut(
                title=chunk.title if chunk else citation.source_url,
                url=citation.source_url,
                retrieved_at=ingested_at.date().isoformat() if ingested_at else None,
            ),
        )

    return rendered


def to_missing_out(plan: GuardedPlan) -> list[MissingItemOut]:
    """The plan's missing list, as the panel's questions. Already filtered by the guard."""
    return [
        MissingItemOut(field_id=item.field_id, label=item.label, question=item.question)
        for item in plan.missing
    ]


def action_log_entries(plan: GuardedPlan) -> list[tuple[str, str, str, str | None]]:
    """
    The `action_log` rows this plan produces: `(action_type, field_id, status, reason)`.

    **Refusals only.** A rejected action is finished — the guard refused it and nothing will
    change that — so `rejected` is the whole truth about it. An approved action has not happened
    yet: it is a proposal the user has still to accept, and writing it as `ok` now would record a
    fill that may never occur. Its execution status arrives with `/results`, which is where the
    spec puts it.

    An action with no field id (a `pause`) is skipped, because `field_id` is NOT NULL and a row
    that named no field would say nothing.
    """
    return [
        (rejected.type, rejected.field_id, "rejected", rejected.code)
        for rejected in plan.rejected
        if rejected.field_id
    ]


def blocked_field_ids(payload: PlanRequest) -> set[str]:
    """
    Every field the content script flagged as a checkpoint, however it said so.

    The two signals are honoured together, not one instead of the other: a field marked
    `sensitive` and a field named in `sensitive_flags` are both blocked, because a page can use
    either and losing one would mean writing into a password box.
    """
    blocked = {flag.field_id for flag in payload.sensitive_flags}
    blocked.update(field.field_id for field in payload.fields if field.sensitive)

    return blocked


def build_response(
    *,
    plan_id: str,
    payload: PlanRequest,
    outcome: PlanTurn,
    step: StepOut | None,
) -> PlanResponse:
    """The preview, assembled from the guarded plan. Pure."""
    plan = outcome.plan
    labels = labels_by_field_id(payload.fields, payload.buttons)

    return PlanResponse(
        plan_id=plan_id,
        cached=False,
        client_plan_id=payload.client_plan_id,
        reply=plan.reply,
        actions=[
            to_action_out(index, action, labels)
            for index, action in enumerate(plan.approved, start=1)
        ],
        rejected=[
            to_rejected_out(rejected, labels) for rejected in plan.rejected
        ],
        missing=to_missing_out(plan),
        citations=to_citations_out(plan.citations, outcome.chunks),
        grounding=plan.grounding,
        step=step,
    )
