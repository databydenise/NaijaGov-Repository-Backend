"""
Text helpers for the renderer: sanitisation, masking, and a token estimate.

Non-business functions, pure and side-effect free. They are the mechanical part of keeping
untrusted page text safe and the user's values hidden; `context.py` decides *where* the results
go, these decide *what a value becomes* on the way in.
"""

import re
from typing import Final

from src.ai.constants import (
    CHARS_PER_TOKEN,
    MASK_MIN_LENGTH,
    OFFICIAL_SOURCES_CLOSE,
    OFFICIAL_SOURCES_CLOSE_ESCAPED,
    OFFICIAL_SOURCES_OPEN,
    OFFICIAL_SOURCES_OPEN_ESCAPED,
    SNAPSHOT_CLOSE,
    SNAPSHOT_CLOSE_ESCAPED,
    SNAPSHOT_OPEN,
    SNAPSHOT_OPEN_ESCAPED,
    USER_DATA_CLOSE,
    USER_DATA_OPEN,
    WORKFLOW_CLOSE,
    WORKFLOW_CLOSE_ESCAPED,
    WORKFLOW_OPEN,
    WORKFLOW_OPEN_ESCAPED,
)

# Control characters (C0 and C1, including newline and tab) become a space; runs of whitespace
# then collapse to one. A label spread over several lines is one line by the time the model sees
# it, and a NUL that would break a log or a terminal is gone.
_CONTROL_CHARS: Final = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WHITESPACE: Final = re.compile(r"\s+")

# Delimiters a hostile page might place in its own text to close a fenced block early and pass
# instructions off as ours. Rewritten to look-alikes that no parser here treats as a delimiter.
_DELIMITER_SWAPS: Final[tuple[tuple[str, str], ...]] = (
    (SNAPSHOT_OPEN, SNAPSHOT_OPEN_ESCAPED),
    (SNAPSHOT_CLOSE, SNAPSHOT_CLOSE_ESCAPED),
    (USER_DATA_OPEN, "‹user_data›"),
    (USER_DATA_CLOSE, "‹/user_data›"),
    # `/explain` fences retrieved corpus text as well. A chunk comes from a government site
    # rather than from a hostile page, but it is still text nobody on this side wrote.
    (OFFICIAL_SOURCES_OPEN, OFFICIAL_SOURCES_OPEN_ESCAPED),
    (OFFICIAL_SOURCES_CLOSE, OFFICIAL_SOURCES_CLOSE_ESCAPED),
    # The registry block. This one matters most of the three: it is the only block the prompt
    # tells the model to *trust*, so a page label that closed it early would be writing with our
    # authority rather than merely adding noise to a block already declared untrustworthy.
    (WORKFLOW_OPEN, WORKFLOW_OPEN_ESCAPED),
    (WORKFLOW_CLOSE, WORKFLOW_CLOSE_ESCAPED),
)


def clean(text: str, limit: int | None = None) -> str:
    """
    Make untrusted text safe to place in a prompt: no control characters, no early fence break,
    one line, and no longer than `limit`. Every value that came from a page passes through here.
    """
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()

    for delimiter, replacement in _DELIMITER_SWAPS:
        text = text.replace(delimiter, replacement)

    if limit is not None and len(text) > limit:
        text = text[:limit]

    return text


def estimate_tokens(text: str) -> int:
    """
    A rough token count from character length. Used only to decide when to drop history and
    non-required fields, so an estimate is enough; a real tokeniser would slot in here.
    """
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def mask_value(value: str) -> str:
    """
    A preview that shows the *kind* of a value without showing the value.

    An email keeps its domain (`a…@example.com`) so the model can tell it from a phone number;
    the rest collapse to the first character, an ellipsis, and the last two — and a short value
    hides even those, so the mask can never reproduce the whole thing. The field key already
    tells the model the type; this is a sanity hint, not data worth exfiltrating.
    """
    value = value.strip()

    if "@" in value:
        _, _, domain = value.partition("@")
        return f"{value[0]}…@{domain}"

    if len(value) <= MASK_MIN_LENGTH:
        return f"{value[0]}…"

    return f"{value[0]}…{value[-2:]}"
