"""
The guard: a parsed model response in, a plan that is safe to show out.

One entry point, `guard_plan`, and it assumes the model may be wrong, confused, or steered by a
hostile page. Everything it can check independently, it checks: the field exists in *this*
snapshot, the field is not one the content script flagged, the type agrees, the value came from
the profile or from something the user typed, the option is really in the list. The per-action
checks live in `checks.py`; what is assembled here is everything that needs the whole list —
uniqueness, volume, grounding and the report, with the merged `missing` list in `missing.py`.

Pure functions: no database, no HTTP, no clock, no model, and no logging. The report it returns is
what P4 logs. That purity is what makes the fragment exhaustively checkable, which is the point of
a last gate.
"""

from collections.abc import Collection, Mapping, Sequence

from src.ai.schemas import PlanResponse
from src.context.schemas import PageButton, PageField
from src.documents.schemas import RetrievedChunk
from src.guard.checks import check_action, reject
from src.guard.constants import (
    BLOCKED_BUTTON_PAUSE_REASON,
    MAX_APPROVED_ACTIONS,
    READ_ONLY_ACTIONS,
    UNVERIFIED_REPLY,
    GroundingVerdict,
)
from src.guard.grounding import grounding_verdict, verify_citations
from src.guard.missing import merge_missing
from src.guard.schemas import ApprovedAction, GuardedPlan, GuardReport, RejectedAction
from src.guard.utils import chat_values_from, comparable


def _dedupe_key(approved: ApprovedAction) -> tuple[str, str]:
    """
    What makes two approved actions the same action.

    A field is written once: `fill`, `select`, `check` and `clickSafe` share a key, so two writes
    to one control cannot both survive whichever types they claim. A read-only action is keyed by
    its type as well, because highlighting a field we also filled is not a duplicate. A `pause` is
    keyed by its reason, so the panel is not handed the same sentence three times.
    """
    if approved.type == "pause":
        return ("pause", comparable(approved.reason or ""))

    if approved.type in READ_ONLY_ACTIONS:
        return (approved.type, approved.field_id or "")

    return ("write", approved.field_id or "")


def _report(
    approved: Sequence[ApprovedAction],
    rejected: Sequence[RejectedAction],
    *,
    grounding: GroundingVerdict,
    citations_verified: int,
    citations_dropped: int,
    missing: int,
    missing_added: int,
    repair_requested: bool,
    reply_replaced: bool,
) -> GuardReport:
    """Counts, codes and field ids. Never a value, never a label, never the reply."""
    codes: dict[str, int] = {}
    for item in rejected:
        codes[item.code] = codes.get(item.code, 0) + 1

    return GuardReport(
        approved=len(approved),
        rejected=len(rejected),
        suspicious=sum(1 for action in approved if action.suspicious),
        rejection_codes=codes,
        rejected_field_ids=tuple(item.field_id for item in rejected if item.field_id),
        grounding=grounding,
        citations_verified=citations_verified,
        citations_dropped=citations_dropped,
        missing=missing,
        missing_added=missing_added,
        repair_requested=repair_requested,
        reply_replaced=reply_replaced,
    )


def guard_plan(
    response: PlanResponse,
    *,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton] = (),
    profile: Mapping[str, str | None],
    chat_values: Mapping[str, str] | None = None,
    chunks: Sequence[RetrievedChunk] = (),
    blocked_field_ids: Collection[str] = (),
    repair_attempted: bool = False,
) -> GuardedPlan:
    """
    Check a model response against the page it was made about, and return a plan safe to show.

    `fields` and `buttons` are this turn's snapshot — the one the request was made against, never
    a stored one. `profile` holds real values; `chat_values` holds what the user typed in earlier
    turns, and this turn's `extracted_data` is merged in. `chunks` are what retrieval returned
    this turn, and a citation to anything else is dropped.

    `blocked_field_ids` is the snapshot's `sensitive_flags` list. It carries the same signal as a
    field's own `sensitive` flag and is honoured as well as it, not instead: either one blocks.

    `repair_attempted` says this is the second attempt, because a pure function cannot know. Only
    then is an ungrounded reply replaced; on a first attempt the guard asks P4 for the repair.
    """
    field_index = {page_field.field_id: page_field for page_field in fields}
    button_index = {button.field_id: button for button in buttons}
    blocked = set(blocked_field_ids)
    chat = chat_values_from(chat_values, response.extracted_data)

    approved: list[ApprovedAction] = []
    rejected: list[RejectedAction] = []
    keys: set[tuple[str, str]] = set()

    for action in response.actions:
        outcome = check_action(action, field_index, button_index, blocked, profile, chat)

        if isinstance(outcome, RejectedAction):
            rejected.append(outcome)
            if outcome.code != "BLOCKED_BUTTON":
                continue
            # A blocked button does not simply disappear: the user is asked to press it.
            outcome = ApprovedAction(type="pause", reason=BLOCKED_BUTTON_PAUSE_REASON)

        # 7. Uniqueness — keep the first, reject the repeat.
        key = _dedupe_key(outcome)
        if key in keys:
            rejected.append(reject(action, "DUPLICATE"))
            continue

        # 8. Volume — the surplus is rejected, not truncated silently.
        if len(approved) >= MAX_APPROVED_ACTIONS:
            rejected.append(reject(action, "TOO_MANY"))
            continue

        keys.add(key)
        approved.append(outcome)

    # The keys that could supply a value, so `merge_missing` does not ask for what we have.
    held = {
        key
        for key in (*profile.keys(), *chat.keys())
        if (profile.get(key) or chat.get(key))
    }
    missing, added = merge_missing(response.missing, fields, approved, blocked, held)

    citations, dropped = verify_citations(response.citations, chunks)
    verdict = grounding_verdict(response.reply, citations)

    reply = response.reply
    reply_replaced = False
    repair_requested = False

    if verdict == "unverified":
        if repair_attempted:
            # Second attempt, still unsupported: the claim goes, the actions stay. A wrong
            # sentence does not invalidate a correct fill. Nothing is cited, so nothing is shown.
            reply = UNVERIFIED_REPLY
            reply_replaced = True
            citations = []
        else:
            repair_requested = True

    return GuardedPlan(
        reply=reply,
        approved=tuple(approved),
        rejected=tuple(rejected),
        citations=tuple(citations),
        missing=tuple(missing),
        grounding=verdict,
        repair_requested=repair_requested,
        reply_replaced=reply_replaced,
        chat_values=chat,
        report=_report(
            approved,
            rejected,
            grounding=verdict,
            citations_verified=len(citations),
            citations_dropped=dropped,
            missing=len(missing),
            missing_added=added,
            repair_requested=repair_requested,
            reply_replaced=reply_replaced,
        ),
    )
