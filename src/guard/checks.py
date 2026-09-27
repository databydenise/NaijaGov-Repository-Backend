"""
The per-action checks, in the order the spec fixes and with the first failure winning.

Order is not a style choice. Cheap structural checks come first, then safety, then the checks
that need a lookup, so a malformed action never reaches value resolution and a sensitive field is
refused before anything is fetched for it. Each check answers one question, and every failure
returns a code from the published list rather than a boolean, because the panel has to tell the
user what happened.

The rule these share: verify, never repair. An id that is not in the snapshot is dropped, not
matched to the nearest similar field. An option that does not appear in the list is refused, not
approximated. A `fill` aimed at a select is refused, not converted. Every one of those "helpful"
alternatives puts a value the user never approved into a government form.
"""

from collections.abc import Collection, Mapping

from src.ai.schemas import PlannedAction, ValueRef
from src.context.schemas import PageButton, PageField
from src.guard.constants import (
    CHECK_FIELD_TYPES,
    FILL_FIELD_TYPES,
    MAX_VALUE_CHARS,
    READ_ONLY_ACTIONS,
    REJECTION_MESSAGES,
    SELECT_FIELD_TYPES,
    RejectionCode,
)
from src.guard.schemas import ApprovedAction, RejectedAction
from src.guard.utils import (
    clean_note,
    clean_value,
    comparable,
    is_suspicious,
    match_option,
)
from src.profiles.constants import PROFILE_FIELDS


def reject(action: PlannedAction, code: RejectionCode) -> RejectedAction:
    """One rejection, carrying the sentence the panel shows for that code."""
    return RejectedAction(
        type=action.type,
        code=code,
        message=REJECTION_MESSAGES[code],
        field_id=action.field_id,
    )


def resolve_value(
    ref: ValueRef,
    profile: Mapping[str, str | None],
    chat: Mapping[str, str],
) -> tuple[str | None, RejectionCode | None]:
    """
    A reference turned into a literal value, or the code that says why not.

    The model only ever sends a key, so this is the single place a real value enters a plan — and
    it can only come from the two stores passed in. There is no third branch and no "the model
    probably meant this": a key we do not hold is `NO_VALUE`, and the panel asks the user for it.
    """
    # The key must be one of the profile's own field names, whichever store it comes from.
    # `extracted_data` is already restricted to that list (P2), and this closes the other end: a
    # reference to `profile.password_hash` resolves to nothing however the caller built the
    # mapping it passed, and a polluted session cannot introduce a key of its own.
    if ref.key not in PROFILE_FIELDS:
        return None, "NO_VALUE"

    store: Mapping[str, str | None] = profile if ref.source == "profile" else chat
    raw = store.get(ref.key)

    if raw is None:
        return None, "NO_VALUE"

    value = clean_value(str(raw))

    if not value:
        return None, "EMPTY_VALUE"

    # The published code list has no length code, and adding one would change what the panel
    # renders, so an absurd value is malformed rather than an outcome of its own.
    if len(value) > MAX_VALUE_CHARS:
        return None, "MALFORMED"

    return value, None


def _check_writable_field(
    action: PlannedAction,
    page_field: PageField,
    profile: Mapping[str, str | None],
    chat: Mapping[str, str],
) -> ApprovedAction | RejectedAction:
    """Checks 4 to 6 for `fill`, `select` and `check`: type, value, option."""
    field_type = comparable(page_field.type)
    rejection: RejectedAction | None = None
    approved: ApprovedAction | None = None

    if action.type == "check":
        if field_type not in CHECK_FIELD_TYPES:
            rejection = reject(action, "MALFORMED")
        else:
            # No `checked` means "tick it": that is what the model asks for when it wants a
            # declaration accepted, and the extension needs a definite boolean either way.
            approved = ApprovedAction(
                type=action.type,
                field_id=action.field_id,
                checked=True if action.checked is None else action.checked,
                note=clean_note(action.note),
            )
    else:
        # 4. A `fill` aimed at a select is rejected, never coerced into a `select`.
        valid_type = (
            action.type == "fill" and field_type in FILL_FIELD_TYPES
        ) or (action.type == "select" and field_type in SELECT_FIELD_TYPES)
        if not valid_type or action.value_ref is None:
            rejection = reject(action, "MALFORMED")
        else:
            # 5. Value resolves, from the profile or from this conversation. Nowhere else.
            value, code = resolve_value(action.value_ref, profile, chat)
            if value is None:
                rejection = reject(action, code or "NO_VALUE")
            elif action.type == "select":
                # 6. Option membership, in the option's own spelling.
                matched = match_option(value, page_field.options)
                if matched is None:
                    rejection = reject(action, "BAD_OPTION")
                else:
                    value = matched

            if rejection is None:
                approved = ApprovedAction(
                    type=action.type,
                    field_id=action.field_id,
                    value=value,
                    source=f"{action.value_ref.source}.{action.value_ref.key}",
                    note=clean_note(action.note),
                    suspicious=is_suspicious(action.value_ref.key, page_field.label),
                )

    if rejection is not None:
        return rejection
    assert approved is not None
    return approved


def _check_pause(action: PlannedAction) -> ApprovedAction | RejectedAction:
    """Validate pause actions without letting the guard logic grow more branches."""
    reason = clean_note(action.reason) or ""
    if not reason:
        return reject(action, "MALFORMED")

    return ApprovedAction(type="pause", reason=reason, note=clean_note(action.note))


def _check_click_safe(
    action: PlannedAction,
    buttons: Mapping[str, PageButton],
    blocked: Collection[str],
) -> ApprovedAction | RejectedAction:
    """Validate safe-click requests and keep the main checker readable."""
    button = buttons.get(action.field_id)  # pyright: ignore[reportArgumentType]
    if button is None:
        return reject(action, "UNKNOWN_FIELD")
    if button.sensitive or action.field_id in blocked:
        return reject(action, "BLOCKED_BUTTON")

    return ApprovedAction(
        type=action.type,
        field_id=action.field_id,
        note=clean_note(action.note),
    )


def _check_read_only(
    action: PlannedAction,
    fields: Mapping[str, PageField],
    buttons: Mapping[str, PageButton],
) -> ApprovedAction | RejectedAction:
    """Allow read-only actions when they point to a known field or button."""
    if action.field_id not in fields and action.field_id not in buttons:
        return reject(action, "UNKNOWN_FIELD")

    return ApprovedAction(
        type=action.type,
        field_id=action.field_id,
        reason=clean_note(action.reason),
        note=clean_note(action.note),
    )


def _check_writable_target(
    action: PlannedAction,
    fields: Mapping[str, PageField],
    blocked: Collection[str],
    profile: Mapping[str, str | None],
    chat: Mapping[str, str],
) -> ApprovedAction | RejectedAction:
    """Apply the remaining field checks for writable actions."""
    page_field = fields.get(action.field_id)  # pyright: ignore[reportArgumentType]
    if page_field is None:
        return reject(action, "UNKNOWN_FIELD")

    if page_field.sensitive or action.field_id in blocked:
        return reject(action, "SENSITIVE_FIELD")

    return _check_writable_field(action, page_field, profile, chat)


def check_action(
    action: PlannedAction,
    fields: Mapping[str, PageField],
    buttons: Mapping[str, PageButton],
    blocked: Collection[str],
    profile: Mapping[str, str | None],
    chat: Mapping[str, str],
) -> ApprovedAction | RejectedAction:
    """
    One action through checks 1 to 6.

    Uniqueness and volume need the whole list and are applied by `service.guard_plan`.
    """
    if action.type == "pause":
        return _check_pause(action)

    if not action.field_id:
        return reject(action, "MALFORMED")

    if action.type == "clickSafe":
        return _check_click_safe(action, buttons, blocked)

    if action.type in READ_ONLY_ACTIONS:
        return _check_read_only(action, fields, buttons)

    return _check_writable_target(action, fields, blocked, profile, chat)
