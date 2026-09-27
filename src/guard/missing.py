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


def merge_missing(
    reported: Sequence[MissingItem],
    fields: Sequence[PageField],
    approved: Sequence[ApprovedAction],
    blocked: Collection[str],
) -> tuple[list[MissingItem], int]:
    """
    The model's list, pruned, plus every required field with no approved write. Returns the list
    and how many entries the guard added itself.

    Sensitive fields are never added: "What is your Password?" is the one question this product
    must not ask, and a checkpoint is the user's to clear, not a gap in our data.
    """
    known = {page_field.field_id for page_field in fields}
    written = {
        action.field_id for action in approved if action.type in WRITE_ACTIONS and action.field_id
    }

    merged: list[MissingItem] = []
    seen: set[str] = set()

    for item in reported:
        if item.field_id not in known or item.field_id in written or item.field_id in seen:
            continue

        seen.add(item.field_id)
        merged.append(item)

    added = 0
    for page_field in fields:
        if not page_field.required or page_field.sensitive or page_field.field_id in blocked:
            continue
        if page_field.field_id in written or page_field.field_id in seen:
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
