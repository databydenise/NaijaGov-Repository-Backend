"""
The guard's codes, limits and fixed sentences.

Two catalogues live here, and both are part of a published contract:

- `REJECTION_MESSAGES` pairs every rejection code with the sentence the side panel shows a
  citizen. The panel renders it verbatim, so it is written for someone who has already lost a
  morning to a portal: plain, unapologetic, and it says what they should do instead. Adding a
  code is a change the extension has to render, not a one-line edit.
- `UNVERIFIED_REPLY` is the sentence that replaces an ungrounded reply. Fixed, never formatted,
  because the one thing worse than "I don't know" is a confident fee that no source contains.

Nothing here is a tuning knob. Every limit below is what stops a confused or steered model
turning one model call into a page full of writes.
"""

from typing import Final, Literal

RejectionCode = Literal[
    "UNKNOWN_FIELD",
    "SENSITIVE_FIELD",
    "BAD_OPTION",
    "NO_VALUE",
    "EMPTY_VALUE",
    "DUPLICATE",
    "TOO_MANY",
    "BLOCKED_BUTTON",
    "MALFORMED",
]

GroundingVerdict = Literal["grounded", "not_required", "unverified"]

# Shown verbatim in the fill preview beside the row that was dropped. Each says what happened
# and, where there is one, what the user can do — never a code, never a field label.
REJECTION_MESSAGES: Final[dict[RejectionCode, str]] = {
    "UNKNOWN_FIELD": "I couldn't find that field on this page, so I left it alone.",
    "SENSITIVE_FIELD": "This field needs you — I don't fill passwords or codes.",
    "BAD_OPTION": "None of the choices in that list matched what I have, so I left it for you.",
    "NO_VALUE": "I don't have that detail yet, so I couldn't fill it in.",
    "EMPTY_VALUE": "What I have for that field is blank, so I left it empty.",
    "DUPLICATE": "I'd already filled that field, so I skipped the repeat.",
    "TOO_MANY": "That was more changes than I do at once, so I stopped and left the rest to you.",
    "BLOCKED_BUTTON": (
        "I don't press this button myself — have a look, then press it when you're ready."
    ),
    "MALFORMED": "I wasn't sure what that suggestion meant, so I didn't apply it.",
}

# What a rejected `clickSafe` turns into, so the user is told to click it rather than the
# suggestion vanishing. The panel shows a `pause` with its reason.
BLOCKED_BUTTON_PAUSE_REASON: Final = REJECTION_MESSAGES["BLOCKED_BUTTON"]

# Replaces the reply after a second attempt is still ungrounded. A fixed literal: no field, no
# topic, nothing interpolated, so it cannot be turned into a claim of its own.
UNVERIFIED_REPLY: Final = (
    "I don't have official guidance on that. Please check with the agency directly."
)

# --- Action limits ---
# Matches `MAX_ACTIONS` in `ai/constants.py`: the model may not return more than 30 and the
# guard may not approve more than 30. Kept as its own name because the guard also counts the
# actions it synthesised (a rejected `clickSafe` becomes a `pause`), which the model did not send.
MAX_APPROVED_ACTIONS: Final = 30

# A profile value is a name, an address, a phone number. 500 characters is far past any of
# those and well short of a paragraph pasted into a form by a model that lost its way.
MAX_VALUE_CHARS: Final = 500

# `reason` and `note` are the model's own prose, shown beside a row in the panel. `PlanResponse`
# caps the reply but not these, so the cap is here: a row in a side panel has space for a
# sentence, and a model that returns an essay gets one sentence's worth of it.
MAX_NOTE_CHARS: Final = 200

# --- Type agreement ---
# `fill` writes text, so it may only target a control that takes text. An allowlist, not a
# denylist: an unrecognised control type is refused rather than written into hopefully.
# `""` is here because the content script sends an empty type for a plain input, which the
# renderer already shows to the model as "text".
FILL_FIELD_TYPES: Final = frozenset(
    {"", "text", "email", "tel", "number", "date", "textarea", "url", "search"},
)
# `select-one` / `select-multiple` are what `HTMLSelectElement.type` reports, so both arrive
# in practice even though the seed and the goldens say "select".
SELECT_FIELD_TYPES: Final = frozenset({"select", "select-one", "select-multiple"})
CHECK_FIELD_TYPES: Final = frozenset({"checkbox", "radio"})

# Actions that change the page. They are unique per field, must pass the writability and
# type checks, and are the only ones that may carry a value.
WRITE_ACTIONS: Final = frozenset({"fill", "select", "check"})
# Actions that only draw attention to something. Allowed on a sensitive field on purpose: the
# model is shown blocked fields so it can explain a refusal, and highlighting a password box
# while saying "this one is yours to type" is the behaviour we want.
READ_ONLY_ACTIONS: Final = frozenset({"highlight", "scroll", "explain"})

# --- The soft `suspicious` check ---
# A key aimed at a field whose label says something else. Deliberately small and one-sided:
# these are pairs that are almost certainly a mistake, not everything that looks odd. A match
# only counts when none of the key's own words (`SELF_WORDS`) appear in the label, so
# "State / LGA" and "Alternative Contact Email" are left alone.
SUSPICIOUS_LABEL_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "email": ("phone", "mobile", "telephone", "password", "address"),
    "phone": ("email", "password", "name"),
    "full_name": ("email", "phone", "password", "address"),
    "address": ("email", "phone", "password"),
    "state": ("local government", "lga"),
    "lga": ("state", "country"),
}
# The words that mean "this label really is about that key". Checked first; if one is present
# the pair is not flagged, however many warning words follow it.
SELF_WORDS: Final[dict[str, tuple[str, ...]]] = {
    "email": ("email", "e mail", "e-mail"),
    "phone": ("phone", "mobile", "telephone", "gsm"),
    "full_name": ("name",),
    "address": ("address", "street"),
    "state": ("state",),
    "lga": ("lga", "local government"),
}
