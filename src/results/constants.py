"""
`/results`' vocabulary: statuses, reason codes, the hint table, and the caps.

Two catalogues here are contracts with the *extension*, not with this service, and both are
two-repo changes rather than one-line edits:

- `RESULT_REASONS` is the content script's own failure vocabulary. Every code in it describes
  something that happened to a DOM element at write time, which is why it is **not**
  `guard/constants.py::RejectionCode` even though four names appear in both lists. A guard's
  `SENSITIVE_FIELD` means "the backend refused to plan a write here"; the content script's means
  "the detector flagged it in the live page a moment before writing". Same name, different
  evidence, different phase — and importing one for the other would make the metrics lie about
  which layer caught what.
- `NEXT_HINTS` is the sentence the panel prints verbatim at the end of a run. None of them says
  the Copilot will press Continue, because it will not.

The hint is a decision table rather than a model call. One sentence is not worth two seconds of
a user's time, and a fixed table is the only version of this that cannot invent a next step.
"""

from typing import Final

# --- Statuses -----------------------------------------------------------------------------
#
# What the content script may report per action. Mirrors `ACTION_STATUSES` in
# `sessions/constants.py`, which is the same list as a database CHECK — named separately here
# because this one is a request contract and that one is a column constraint, and a change has
# to be made deliberately in both.
RESULT_STATUSES: Final[tuple[str, ...]] = ("ok", "changed", "failed", "rejected", "cancelled")

# Statuses that mean the user still has something to do on this page, which is what turns the
# hint into `FIX_FIRST`. `cancelled` is absent: those actions never ran, and the reason they did
# not is already being reported as a checkpoint or an abort.
UNFINISHED_STATUSES: Final[frozenset[str]] = frozenset({"failed", "rejected"})

# --- Reason codes -------------------------------------------------------------------------
#
# The content script's vocabulary, per its own execution-engine spec. Validated rather than
# accepted as free text: an unrecognised code is either a version skew worth catching in
# development or a client sending us prose, and prose is how a value ends up in this column.
RESULT_REASONS: Final[frozenset[str]] = frozenset(
    {
        # Batch-level: the run never really started, or stopped for the whole page.
        "STALE_PAGE",
        "CHECKPOINT",
        "NAVIGATED",
        "TIMEOUT",
        "CANCELLED",
        # Per-action, in the order the content script checks them.
        "MALFORMED",
        "UNKNOWN_FIELD",
        "DETACHED",
        "NOT_VISIBLE",
        "NOT_WRITABLE",
        "SENSITIVE_FIELD",
        "WRONG_TYPE",
        "MANUAL_FIELD",
        "BAD_OPTION",
        "NOT_ACCEPTED",
    },
)

# Why a whole batch stopped, as the request may report it. Narrower than `RESULT_REASONS`: these
# three are the only aborts the engine can decide on its own, and a checkpoint is reported in its
# own field rather than as an abort because it is the one that is not a malfunction.
ABORT_REASONS: Final[frozenset[str]] = frozenset({"stale_page", "navigated", "timeout"})

# --- Request caps -------------------------------------------------------------------------
#
# Thirty results, matching the thirty actions a plan may carry (`ai/constants.py::MAX_ACTIONS`
# and the guard's own cap). A report longer than the plan it reports on is not a report.
MAX_RESULTS: Final = 30

# `a1`, `a17` — positional ids this service minted itself in the plan response.
MAX_ACTION_ID_CHARS: Final = 32

# A uuid4 hex from `plan.store.new_plan_id()` is 32 characters. Capped at twice that so a future
# id format has room, and capped at all because it is a string from a client that reaches a
# database key.
MAX_PLAN_ID_CHARS: Final = 64

# A code from the catalogue above, with room for one the extension adds before this list does.
MAX_REASON_CHARS: Final = 40

# Ten minutes in milliseconds. The engine's own batch budget is ten seconds; this is two orders
# of magnitude above it, so it refuses a nonsense number without arguing about a slow machine.
MAX_ELAPSED_MS: Final = 600_000

# Keys that would carry what the user typed, or what the page said. Built from `/context`'s list
# so the two cannot drift apart on `value`, plus the two this endpoint's own shape invites —
# a `label` beside a field id, and the `text` that was written.
#
# This is the endpoint where someone will eventually think it would be handy to log what was
# filled. The tripwire is here so that thought fails a request instead of shipping.
FORBIDDEN_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {"value", "values", "text", "text_content", "label", "labels"},
)

# --- Rate limiting ------------------------------------------------------------------------
#
# Sixty a minute per token — generous, as the spec asks, because this is a reporting call that
# fires once per run and a client that retries a dropped report must not be punished for it.
# Three times `/plan`'s limit, and the same as `/context`'s.
RATE_LIMIT: Final = 60
RATE_WINDOW_SECONDS: Final = 60

# --- The next-step hint -------------------------------------------------------------------

CHECKPOINT_PENDING: Final = "CHECKPOINT_PENDING"
FIX_FIRST: Final = "FIX_FIRST"
FINAL_REVIEW: Final = "FINAL_REVIEW"
REVIEW_CONTINUE: Final = "REVIEW_CONTINUE"

# Printed verbatim in the panel, so each is written for a citizen who has just watched a form
# fill itself and wants to know what they do now. The last two are the only ones that mention
# Continue or Submit, and both put the click on the user.
NEXT_HINTS: Final[dict[str, str]] = {
    CHECKPOINT_PENDING: "Finish that step, then I'll pick up where we left off.",
    FIX_FIRST: "A few fields still need you before moving on.",
    FINAL_REVIEW: "This is the last step. Check everything, then submit it yourself.",
    REVIEW_CONTINUE: "Looks good. Review it, then click Continue yourself.",
}

# What an unacknowledged report is answered with. The run is not ours to comment on — we have no
# record of the plan — so the hint claims nothing about the page and puts the next move on the
# user, which is true whatever happened.
DEFAULT_HINT_CODE: Final = REVIEW_CONTINUE

# --- Checkpoint kinds ---------------------------------------------------------------------
#
# The kinds a checkpoint may be counted under. The *event* row keeps whatever short string the
# extension sent, because that is per-run detail and it cascades away with the session; the
# *counter* is normalised to one of these, because "how often does each kind fire" is not a
# question free text can answer — a hundred spellings of one kind is a hundred rows that each
# look rare.
#
# Normalising rather than validating is deliberate: a kind this list has not caught up with yet
# is counted as `other` and the report is still accepted. Refusing it would lose a whole run's
# results over a word.
CHECKPOINT_KINDS: Final[frozenset[str]] = frozenset(
    {"password", "one_time_code", "captcha", "payment"},
)

# Spellings that mean one of the kinds above. The panel, the content script and this service
# have each had their own wording for the OTP case at some point, and a counter keyed on the
# spelling would have split it three ways.
CHECKPOINT_KIND_ALIASES: Final[dict[str, str]] = {
    "otp": "one_time_code",
    "one time code": "one_time_code",
    "onetime code": "one_time_code",
    "verification code": "one_time_code",
    "sms code": "one_time_code",
    "recaptcha": "captcha",
    "card": "payment",
    "pin": "payment",
}

# What an unrecognised kind is counted as. Present in the numbers rather than dropped: a rising
# `other` is how you find out the extension has started reporting a kind this list should know.
CHECKPOINT_KIND_OTHER: Final = "other"

# --- Aggregate counters -------------------------------------------------------------------
#
# The `metric` column's two namespaced forms. A bare status needs no prefix; these do, so that
# "how often does a password checkpoint fire" and "how often does a fill fail" can never collide
# in one key.
CHECKPOINT_METRIC_PREFIX: Final = "checkpoint:"
ABORT_METRIC_PREFIX: Final = "abort:"

# Stands in for a workflow or step a report has none of — a page outside the registry. A fixed
# marker rather than NULL, because Postgres treats NULLs as distinct in a unique index and the
# counter upsert would insert a new row per report instead of incrementing one.
UNKNOWN_SCOPE_PART: Final = "-"

# What `/health` reports, and how long it is held. Polled, and counting a table only this
# endpoint writes to, so a short cache keeps a liveness check cheap.
COUNTERS_CACHE_KEY: Final = "results:counters"
COUNTERS_CACHE_TTL_SECONDS: Final = 300

# How many rows the `/health` block will summarise. It is a table, not a dashboard: enough to
# show today's shape of things, and bounded so a year of counters cannot become a 2 MB response.
MAX_HEALTH_COUNTERS: Final = 50
