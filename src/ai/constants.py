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

# --- The explain response ---
# Smaller than a plan's, and separately capped rather than sharing `MAX_REPLY_CHARS`: an
# explanation is two or three sentences about one field, and the panel shows it in a card rather
# than in the chat column. A model that writes six sentences fails validation and is repaired once.
MAX_EXPLANATION_CHARS: Final = 400

# One example value — "08012345678", "Sole Proprietor". Eighty characters is a value, not a
# sentence; anything longer is a second explanation wearing an example's clothes.
MAX_EXAMPLE_CHARS: Final = 80

# Three sources on one field. A field explained by five sources is a field the corpus covers
# loosely, and showing all five in a side panel is noise the user has to read past.
MAX_EXPLAIN_CITATIONS: Final = 3

# --- Official sources block (the `/explain` context) ---
# `/explain` retrieves *before* it calls the model, so the sources arrive in the prompt rather
# than through a tool result. Fenced and escaped like the page snapshot: corpus text is copied
# from a government website, which makes it evidence, not instructions.
OFFICIAL_SOURCES_OPEN: Final = "<official_sources>"
OFFICIAL_SOURCES_CLOSE: Final = "</official_sources>"
OFFICIAL_SOURCES_OPEN_ESCAPED: Final = "‹official_sources›"
OFFICIAL_SOURCES_CLOSE_ESCAPED: Final = "‹/official_sources›"

# What one chunk contributes to an explain prompt. Three chunks at 1200 characters is about
# 900 tokens, which leaves the whole explain context an order of magnitude inside `TOKEN_BUDGET`
# — one field and no history is a small prompt by construction, so there is no overflow stage.
MAX_RENDERED_CHUNKS: Final = 3
MAX_RENDERED_CHUNK_CHARS: Final = 1200

# The instruction text beside the field, as the spec caps it, and the user's own question.
MAX_RENDERED_NEARBY_CHARS: Final = 300
MAX_RENDERED_QUESTION_CHARS: Final = 500

# What the model is asked when the user clicked Explain without typing anything. A fixed string
# rather than an empty question, because a prompt whose last line is blank invites the model to
# decide for itself what was being asked.
DEFAULT_EXPLAIN_QUESTION: Final = "What is this field asking for?"

# --- Page block ---
# A long government form is ~40 fields; 80 leaves room for a multi-step page rendered at once
# while still refusing a page built to bury the model in controls. Past it, the block says how
# many were dropped rather than pretending the page is small.
MAX_PAGE_FIELDS: Final = 80
# A label is a few words. Longer is a paragraph sitting in a <label>, and not what helps here.
MAX_RENDERED_LABEL_CHARS: Final = 120
# Enough to show the shape of a choice list without letting one <select> dominate the prompt.
MAX_RENDERED_OPTIONS: Final = 20
# A link's href, rendered so the model can tell a page of this portal from a payment processor.
# Origin and path only by the time it arrives; this caps what a very long path contributes.
MAX_RENDERED_HREF_CHARS: Final = 120

# --- Workflow block ---
# The registry's own view of where the user is. A Nigerian portal workflow is a handful of steps;
# twenty is far past any we would seed and stops a mis-seeded workflow filling the prompt.
MAX_RENDERED_STEPS: Final = 20
# What each step asks for, so "what will I need next" is answerable. Capped per step because the
# point is to tell the user what is coming, not to reproduce a form they cannot see yet.
MAX_RENDERED_STEP_LABELS: Final = 8

# The registry block's fence. Outside `<page_snapshot>` on purpose: this is our own hand-curated
# data, not text copied from a page, and the two must not look alike to the model.
WORKFLOW_OPEN: Final = "<workflow>"
WORKFLOW_CLOSE: Final = "</workflow>"
# Page text that forges this fence is the one injection this block newly makes possible: the
# `<workflow>` block is presented to the model as *ours* and trustworthy, so a label that closed
# it early could pass instructions off as the registry's. Escaped like every other delimiter.
WORKFLOW_OPEN_ESCAPED: Final = "‹workflow›"
WORKFLOW_CLOSE_ESCAPED: Final = "‹/workflow›"

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

# --- The model call (see client.py) ---
# Low, not zero. Zero is not deterministic for a chat model anyway, and a little slack reads
# better in the reply while the parts that must not vary — ids, keys, the schema — are fixed by
# validation rather than by temperature.
MODEL_TEMPERATURE: Final = 0.1

# A ceiling on one completion, so a model that loses its way costs one truncated answer rather
# than a long one. A full plan — a 600-character reply, thirty actions, five citations — is well
# under this; a completion that hits it comes back with `finish_reason="length"`, fails to parse,
# and takes the repair pass.
MAX_COMPLETION_TOKENS: Final = 2000

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
