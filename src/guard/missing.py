"""
Spec Section 4: the merged `missing` list.

The model usually remembers to say what it does not have. This makes it certain. Every required
field the plan left empty becomes a question the panel can ask, whether the model mentioned it or
not, which is the difference between "I don't have your LGA" happening reliably and happening most
of the time.

It also prunes what the model reported, on the same principle as everywhere else in the guard: an
entry naming a field that is not on this page, or a field we just filled, would have the panel ask
the user for something it should not.
"""

from collections.abc import Collection, Sequence

from src.ai.constants import MAX_MISSING
from src.ai.schemas import MissingItem
from src.context.schemas import PageField
from src.guard.constants import WRITE_ACTIONS
from src.guard.schemas import ApprovedAction
from src.guard.utils import clean_value, question_for_field
from src.knowledge.normalize import profile_key_for_label


def merge_missing(
    reported: Sequence[MissingItem],
    fields: Sequence[PageField],
    approved: Sequence[ApprovedAction],
    blocked: Collection[str],
    held: Collection[str] = (),
) -> tuple[list[MissingItem], int]:
    """
    The model's list, pruned, plus every required field we have no value for. Returns the list and
    how many entries the guard added itself.

    `held` is the profile and chat keys that actually have a value. Without it this function asked
    for things we already hold: a turn that answers a question produces no write actions, so every
    required field looked empty and the panel was handed "What is your Email Address?" beside a
    profile that has one. A live turn did exactly that for five of six fields.

    Sensitive fields never survive, whether the guard would have added one or the model reported
    it: "What is your Password?" is the one question this product must not ask, and a checkpoint is
    the user's to clear, not a gap in our data.
    """
    index = {page_field.field_id: page_field for page_field in fields}
    known = set(index)
    written = {
        action.field_id for action in approved if action.type in WRITE_ACTIONS and action.field_id
    }

    merged: list[MissingItem] = []
    seen: set[str] = set()

    for item in reported:
        if item.field_id not in known or item.field_id in written or item.field_id in seen:
            continue

        # The same sensitivity filter the generated entries get below, applied to the model's own
        # list. Without it the rule holds only for what the guard adds, and a model that puts the
        # Password field in its `missing` list has the panel ask for it anyway — which a live turn
        # did, with the field flagged sensitive *and* named in `blocked`.
        page_field = index.get(item.field_id)
        if item.field_id in blocked or (page_field is not None and page_field.sensitive):
            continue

        seen.add(item.field_id)
        merged.append(item)

    added = 0
    for page_field in fields:
        if not page_field.required or page_field.sensitive or page_field.field_id in blocked:
            continue
        if page_field.field_id in written or page_field.field_id in seen:
            continue

        # A field whose value we hold is not a gap in our data, however this turn went. The label
        # is matched through B8's normaliser, so "LGA" and "E-mail Address:" resolve like the
        # canonical labels; anything the map does not know is treated as held-nothing and asked
        # for, which is the safe way round.
        key = profile_key_for_label(page_field.label)
        if key is not None and key in held:
            continue

        seen.add(page_field.field_id)
        merged.append(
            MissingItem(
                field_id=page_field.field_id,
                label=clean_value(page_field.label),
                question=question_for_field(page_field),
            ),
        )
        added += 1

    # The panel cannot ask forty questions. The model's own entries are kept first, so what a cap
    # drops is the tail of the generated ones.
    return merged[:MAX_MISSING], added
