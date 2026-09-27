"""
Limits and fixed strings for the AI contract.

The caps here protect the model call, not a request body (that is `context/constants.py`).
A snapshot has already passed `/context`'s validation by the time it reaches the renderer;
these are the tighter limits that keep one prompt from growing past what the budget — and the
user's patience on mobile data — can afford.
"""

from typing import Final

# --- Response schema limits (enforced by Pydantic on parse, see schemas.py) ---
# The provider's strict schema cannot express a maximum length or count, so these are the
# real enforcement: P4 parses the model's output through `PlanResponse`, and anything over a
# limit fails validation and triggers the single repair pass.
MAX_REPLY_CHARS: Final = 600
MAX_ACTIONS: Final = 30
MAX_CITATIONS: Final = 5
MAX_MISSING: Final = 10

# --- Page block ---
# A long government form is ~40 fields; 80 leaves room for a multi-step page rendered at once
# while still refusing a page built to bury the model in controls. Past it, the block says how
# many were dropped rather than pretending the page is small.
MAX_PAGE_FIELDS: Final = 80
# A label is a few words. Longer is a paragraph sitting in a <label>, and not what helps here.
MAX_RENDERED_LABEL_CHARS: Final = 120
# Enough to show the shape of a choice list without letting one <select> dominate the prompt.
MAX_RENDERED_OPTIONS: Final = 20

# --- Profile block ---
# What a masked short value collapses to: first character, ellipsis, last two characters. A
# value at or below this length hides everything but the first character instead, so the
# mask can never reproduce the whole value.
MASK_MIN_LENGTH: Final = 4

# --- History ---
# The last few turns are context; older ones are dropped rather than summarised, because
# summarising costs a model call this budget does not have.
MAX_HISTORY_TURNS_RENDERED: Final = 6
MAX_HISTORY_TURN_CHARS: Final = 500

# --- Budget ---
# A rough character-per-token ratio for English. This is only used to decide when to drop
# history and non-required fields, so an estimate is enough; if the boundary ever needs to be
# exact, a tokeniser (e.g. tiktoken) would replace `estimate_tokens` without touching callers.
CHARS_PER_TOKEN: Final = 4
# The whole rendered context (page + user data + history + message) must fit under this. On
# overflow the renderer drops history, then non-required fields, and logs every drop.
TOKEN_BUDGET: Final = 6000

# --- Snapshot fencing ---
# Page content is fenced in these delimiters and declared untrusted in the system prompt. Any
# occurrence of a delimiter inside page text is escaped before rendering, so a hostile page
# cannot close the block early and write instructions that read as ours.
SNAPSHOT_OPEN: Final = "<page_snapshot>"
SNAPSHOT_CLOSE: Final = "</page_snapshot>"
# What a delimiter in page text is rewritten to. Recognisable to a reader, inert to a parser.
SNAPSHOT_OPEN_ESCAPED: Final = "‹page_snapshot›"
SNAPSHOT_CLOSE_ESCAPED: Final = "‹/page_snapshot›"

USER_DATA_OPEN: Final = "<user_data>"
USER_DATA_CLOSE: Final = "</user_data>"
