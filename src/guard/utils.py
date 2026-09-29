"""
Mechanics for the guard: cleaning values, matching options, wording questions, and chat values.

Pure and boring on purpose. The one rule that runs through all of it is that a value is
*cleaned*, never *corrected*. Control characters and stray whitespace go, because they are
artefacts of how the value was stored rather than something the user meant to type. Everything
else stays exactly as they gave it: no phone reformatting, no case fixing, no date rewriting. A
portal that wants `0800 000 0000` instead of `08000000000` should say so in a validation message
the user can read, not have us guess at its format and be wrong quietly.

`clean` in `ai/utils.py` deliberately is not reused here: it also rewrites prompt fence
delimiters, which is right for text going *into* a prompt and wrong for a value going into a
government form.
"""

import re
from collections.abc import Collection, Mapping
from typing import Final

from pydantic import BaseModel

from src.context.schemas import PageField
from src.guard.constants import (
    CHECK_FIELD_TYPES,
    MAX_NOTE_CHARS,
    SELF_WORDS,
    SUSPICIOUS_LABEL_WORDS,
)

# C0 and C1 controls, newline and tab included, become a space; the run then collapses. A value
# that arrived with a newline in it is one line by the time it is typed into a field.
_CONTROL_CHARS: Final = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WHITESPACE: Final = re.compile(r"\s+")


def clean_value(raw: str) -> str:
    """A resolved value, ready to be written: no control characters, no stray whitespace."""
    text = _CONTROL_CHARS.sub(" ", raw)

    return _WHITESPACE.sub(" ", text).strip()


def clean_note(raw: str | None) -> str | None:
    """
    A model-written `reason` or `note`, ready to show: cleaned, capped, and None when empty.

    This text is the model's, not the user's, and it reaches the panel unedited otherwise. The cap
    is the guard's because `PlanResponse` has none for these two fields — a limit on the reply does
    not stop thirty actions each carrying a paragraph.
    """
    if raw is None:
        return None

    return clean_value(raw)[:MAX_NOTE_CHARS] or None


# An id we are willing to delete from the model's prose. It must look machine-made — one token of
# letters, digits, hyphens and underscores, with at least one digit in it — because the content
# script chooses these strings and nothing stops one being an ordinary word. `g1-f2` and `f7`
# qualify; a hypothetical id of `email` does not, and survives. That is the safe direction to
# fail: an id that reads as English is a word the user can read, while deleting every occurrence
# of "email" from an answer about an email field would mangle it.
_ID_LIKE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SPACE_BEFORE_PUNCTUATION: Final = re.compile(r"\s+([.,;:!?])")
_EMPTY_ASIDE: Final = re.compile(r"\s*[(\[]\s*[)\]]")


def strip_field_ids(text: str, field_ids: Collection[str]) -> tuple[str, int]:
    """
    The reply with our own field ids taken out of it, and how many occurrences went.

    A field id is this service's internal handle for a control. It means nothing to a citizen, it
    reads like an error code, and it appeared in a live reply inside a sentence the user was meant
    to act on: *select the "Renew Licence" option (g1-f2)*. Ids belong in `actions[].field_id`,
    where the extension resolves them, and nowhere a person reads.

    The prompt asks for this too. This is the enforcement, because a prompt is a request and the
    panel renders whatever comes back. A parenthesised id is removed with its brackets, since it
    is an aside that adds nothing once the id is gone; a bare one is removed on its own.
    """
    removable = sorted(
        (
            field_id
            for field_id in set(field_ids)
            if _ID_LIKE.fullmatch(field_id) and any(char.isdigit() for char in field_id)
        ),
        key=len,
        reverse=True,
    )

    if not removable:
        return text, 0

    alternation = "|".join(re.escape(field_id) for field_id in removable)
    removed = 0

    def drop(_match: re.Match[str]) -> str:
        nonlocal removed
        removed += 1

        return ""

    # Longest-first alternation, so `f1` cannot eat the front of `f12` and leave a stray `2`.
    cleaned = re.sub(rf"\s*[(\[]\s*(?:{alternation})\s*[)\]]", drop, text)
    cleaned = re.sub(
        rf"(?<![A-Za-z0-9_-])(?:{alternation})(?![A-Za-z0-9_-])",
        drop,
        cleaned,
    )

    if not removed:
        return text, 0

    # Tidy what removal left behind: an empty bracket pair, a doubled space, a space before a
    # full stop. Nothing here rewrites a word — only whitespace and now-empty punctuation moves.
    cleaned = _EMPTY_ASIDE.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)

    return cleaned.strip(), removed


def comparable(text: str) -> str:
    """
    The form two strings are compared in: case-folded, whitespace collapsed, nothing else.

    Used for option membership and for reading a label. Not `normalize_label` from
    `knowledge/`: that strips punctuation and applies aliases, which is right for matching a
    label against the registry and wrong for deciding that a value *is* one of a select's
    options. "N/A" and "NA" are different options in a list that offers both.
    """
    return _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", text)).strip().casefold()


def match_option(value: str, options: list[str]) -> str | None:
    """
    The option this value means, in the option's own spelling, or None.

    Exactly one comparable match wins and its original text is returned, so the portal receives
    the string its own markup uses. Zero matches and ambiguous matches both fail: no fuzzy
    matching, no closest match, no prefix. Picking "Lagos Island" because the user said "Lagos"
    is how a form ends up saying something the user never said.
    """
    wanted = comparable(value)
    matches = [option for option in options if comparable(option) == wanted]

    if len(matches) != 1:
        return None

    return matches[0]


def is_suspicious(key: str, label: str) -> bool:
    """
    True when a reference's key and a field's label clearly disagree.

    Soft by design: the action is still approved and merely flagged, because the alternative
    rejects legitimate cases — a contact email belongs in "Alternative Contact". The label is
    only read for words we listed; anything we have no opinion about is not flagged.
    """
    warning_words = SUSPICIOUS_LABEL_WORDS.get(key)
    if not warning_words:
        return False

    text = comparable(label)
    if not text:
        return False

    # The label agreeing with the key wins outright, so "State / LGA" is never flagged.
    if any(word in text for word in SELF_WORDS.get(key, ())):
        return False

    return any(word in text for word in warning_words)


def question_for_field(page_field: PageField) -> str:
    """
    The question the panel asks for a required field the plan has no value for.

    Generated from the label rather than the id, because the user reads it. A checkbox is asked
    about differently: "What is your Declaration?" is not a question anyone can answer.
    """
    label = clean_value(page_field.label)

    if not label:
        return "What should go in this field?"

    if comparable(page_field.type) in CHECK_FIELD_TYPES:
        return f"Can you confirm: {label}?"

    return f"What is your {label}?"


def chat_values_from(stored: Mapping[str, str] | None, extracted: BaseModel) -> dict[str, str]:
    """
    The `chat.*` values a reference may resolve against.

    What the session kept from earlier turns, plus what the model extracted from this turn's
    message, with this turn winning.

    `extracted_data` is a submodel with one nullable field per profile key (P2), and a `null`
    means "not provided" — skipping those is why this exists rather than a dict update. Its values
    are the model's reading of what the user just typed, which is the only way a value given in
    chat can be filled in the same turn it was given.
    """
    values: dict[str, str] = {key: value for key, value in (stored or {}).items() if value}

    for key, value in extracted.model_dump().items():
        if isinstance(value, str) and value.strip():
            values[key] = value

    return values
