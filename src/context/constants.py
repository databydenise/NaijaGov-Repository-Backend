"""
`/context` limits and fixed strings.

The caps are not tuning knobs. They are what stops a hostile or broken page turning one
request into a large amount of work, and every one of them returns `INVALID_REQUEST`
rather than being silently truncated — a snapshot we quietly cut in half would be matched
against the registry as though it were the whole page.
"""

from typing import Final

# A long government form is perhaps 40 fields. 300 leaves room for a page that renders
# every step at once, and refuses one built to make us do work.
MAX_FIELDS: Final = 300
MAX_BUTTONS: Final = 100
MAX_OPTIONS: Final = 200

# A label is a few words. Anything longer is a paragraph that happens to sit in a <label>,
# and it is not what the matcher compares.
MAX_LABEL_LENGTH: Final = 200
MAX_HEADINGS: Final = 20
MAX_HEADING_LENGTH: Final = 300
MAX_URL_LENGTH: Final = 2048
MAX_TITLE_LENGTH: Final = 300

# The client's own hash. Only ever compared, never trusted or stored.
MAX_PAGE_HASH_LENGTH: Final = 128

# Field ids come from the content script and are echoed back in actions. They are short.
MAX_FIELD_ID_LENGTH: Final = 100
MAX_SENSITIVE_FLAGS: Final = MAX_FIELDS

# 256 KB. A 300-field snapshot with 200 options each is well inside this; a body past it is
# not a page we can help with.
MAX_BODY_BYTES: Final = 256 * 1024

# `MutationObserver` on a busy portal fires far more often than a person changes page, so
# the ceiling is set for a machine, not a human.
RATE_LIMIT: Final = 60
RATE_WINDOW_SECONDS: Final = 60

# How long an unchanged page is answered from the existing session without re-matching.
CACHE_WINDOW_SECONDS: Final = 60

# Keys that would carry what a user typed. Rejected by name so the mistake shows up the
# first time someone adds one, not in a review six weeks later.
FORBIDDEN_FIELD_KEYS: Final = frozenset({"value", "values", "text_content"})

# Shown verbatim in the panel, so it is written for a citizen: says what is true, and
# offers the thing that still works.
UNSUPPORTED_MESSAGE: Final = (
    "I don't know this page yet, but I can still answer questions about it."
)
