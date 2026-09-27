"""
`/explain`'s limits, budgets, and the copy a citizen reads.

Most of this file is fixed sentences, and that is the feature rather than an accident of style.
Three of the four answers this endpoint can give are written here rather than generated:

- `CANNED_EXPLANATIONS` — a sensitive field. Deterministic and instant, and the exact sentence you
  want on stage at the moment the Copilot visibly stops doing something.
- `NO_GUIDANCE_EXPLANATION` — the corpus does not cover the field. The honest answer is also the
  cheap one here, which is the whole design.
- `RETRIEVAL_UNAVAILABLE_EXPLANATION` — the lookup did not run. Deliberately *not* the sentence
  above: "no official source covers this" is a claim about the corpus, and a database that did not
  answer is not evidence for it.

Only the fourth answer costs a model call.
"""

from typing import Final

from src.agent.constants import TURN_BUDGET_SECONDS

# --- Request ------------------------------------------------------------------------------
#
# The portal's instruction text beside the field, as the spec caps it. The most useful sentence on
# the page for explaining a field, and the least trustworthy, so it is fenced and escaped like
# every other piece of page text.
MAX_NEARBY_TEXT_CHARS: Final = 300

# What the user may ask about the field instead of "what is this?". Shorter than `/plan`'s 2000:
# this is a question about one control, and a question that long is a conversation, which is what
# `/plan` is for.
MAX_QUESTION_CHARS: Final = 500

# 64 KB. One field — even a `<select>` with two hundred options — plus a question and a little
# nearby text is far inside this. The snapshot endpoints need 256 KB; this one takes a single
# control, and a body past this is not a field we can explain.
MAX_BODY_BYTES: Final = 64 * 1024

# --- Rate limiting ------------------------------------------------------------------------
#
# Thirty a minute per token, as the spec sets it. Higher than `/plan`'s twenty because clicking
# Explain on six fields in a row is a person reading a form, not a loop — and most of those clicks
# are answered from the cache or from fixed copy without a model call at all.
RATE_LIMIT: Final = 30
RATE_WINDOW_SECONDS: Final = 60

# --- Budgets ------------------------------------------------------------------------------
#
# The whole request, measured from the moment the handler starts. Only ever *tightens* the runner's
# own budget: it is what stops a slow retrieval turning a 20-second turn into a request the panel
# has already given up on.
REQUEST_BUDGET_SECONDS: Final = TURN_BUDGET_SECONDS + 3.0

# Held back from the turn's deadline for the cache write that follows it. Small, because the write
# is one upsert — but a turn that spent every last second and then had no time to cache its answer
# would make the second click as slow as the first, which is the one thing this endpoint promises.
CACHE_RESERVE_SECONDS: Final = 1.0

# --- Caching ------------------------------------------------------------------------------
#
# A week, as the spec sets it. Invalidation is otherwise by construction: the key carries the
# prompt version and the agency's ingestion timestamp, so a re-ingest or a prompt change makes
# every old row unreachable rather than needing a purge. This expiry is the backstop for the case
# neither covers — a source that changed on the agency's website and has not been re-ingested.
CACHE_TTL_SECONDS: Final = 7 * 24 * 60 * 60

# What a key's parts are joined with. A character that cannot appear in a normalised label, a
# workflow id or a timestamp, so two different keys cannot collide by rearranging their parts.
CACHE_KEY_SEPARATOR: Final = "\x1f"

# The stamp in a cache key when the corpus has no ingest date to offer — an empty corpus, or a
# lookup that failed. Distinct from a real timestamp, so an answer written while the corpus was
# unreadable is not served after it comes back.
UNKNOWN_CORPUS_STAMP: Final = "corpus-unknown"

# --- Retrieval ----------------------------------------------------------------------------
#
# Three chunks, matching what the prompt can show and what a citation list may name. More would be
# retrieved, rendered, and then uncitable.
SEARCH_LIMIT: Final = 3

# --- The sensitive kinds ------------------------------------------------------------------

# Which canned answer a sensitive field gets. The request carries only `sensitive: bool`, so the
# kind is worked out from the label and the type — in this order, because a field can trip more
# than one list and the more specific reading is the right one: a `type="password"` box labelled
# "One-Time Code" is an OTP box, not a password.
SENSITIVE_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "one_time_code",
        (
            "one-time",
            "one time",
            "onetime",
            "otp",
            "verification code",
            "confirmation code",
            "authentication code",
            "sms code",
            "2fa",
        ),
    ),
    ("captcha", ("captcha", "recaptcha", "robot", "security image")),
    (
        "payment",
        (
            "card number",
            "cardnumber",
            "cvv",
            "cvc",
            "expiry",
            "expiration",
            "account number",
            "payment",
            "transaction pin",
            "remita",
            "paystack",
        ),
    ),
    ("password", ("password", "passphrase", "passcode", "pin")),
)

# Shown verbatim in the panel. Each one says what the field is, that the user does it themselves,
# and why — because "I can't help with this" without a reason reads as a broken tool, and this is
# the moment the product most needs to look deliberate.
CANNED_EXPLANATIONS: Final[dict[str, str]] = {
    "password": (
        "This is the password for your account on this portal. Type it yourself — I never handle "
        "passwords, codes or payments. Use something you haven't used on another site, and keep a "
        "copy somewhere safe."
    ),
    "one_time_code": (
        "This is a one-time code sent to your phone or email. Type it yourself — I never handle "
        "codes, passwords or payments. Codes expire after a few minutes, so ask for a new one if "
        "it stops working."
    ),
    "captcha": (
        "This checks that a person, not a program, is filling the form. You'll need to solve it "
        "yourself — I can't, and I won't try. If the image is unreadable, most portals offer a "
        "refresh or an audio version."
    ),
    "payment": (
        "This is a payment detail, so it stays between you and the portal — I never handle card "
        "numbers, account details or PINs. Check the page is the agency's own before you type it."
    ),
}

# For a field the content script marked sensitive that matches none of the lists above. Claims
# nothing about what the field is, because we do not know — only that it is the user's to fill.
GENERIC_SENSITIVE_EXPLANATION: Final = (
    "This one is yours to fill in. Passwords, codes and payment details stay between you and the "
    "portal, so I don't touch a field the page marks as private."
)

# --- The answers that cost nothing --------------------------------------------------------

# The spec's own sentence, and it does two things: it says plainly that we do not know, and it
# names somewhere that does. "No rule, no claim" is only usable advice if it ends with a next step.
NO_GUIDANCE_EXPLANATION: Final = (
    "I don't have official guidance for this field. The portal's help page or the agency's "
    "support line will know."
)

# Not the sentence above, and the difference matters more than it reads. An empty search result is
# evidence that no official source covers the field; a lookup that did not run is evidence of
# nothing at all, and reporting it as the former is how a service starts inventing absences.
RETRIEVAL_UNAVAILABLE_EXPLANATION: Final = (
    "I can't check the official sources at the moment, so I'd rather not guess about this field. "
    "Try again in a moment."
)

# A step line for a page with no workflow, matching `/plan`'s. A session that has none is normal:
# `/context` writes one for any page outside the registry, and the panel still offers Explain there.
UNKNOWN_STEP_NAME: Final = "This page"

# What stands in for a workflow or a step in a cache key when the page has neither. A fixed marker
# rather than an empty string, so a key's shape is the same whether or not the page is known.
UNKNOWN_SCOPE_PART: Final = "-"
