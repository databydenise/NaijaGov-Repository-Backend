"""
Matching thresholds, in one place.

Every number here decides whether the panel acts on its own or asks. They are set for a
six-field and a three-field step, which is what the registry holds; revisit them against
real pages rather than tuning them to make one stubborn case pass.
"""

from typing import Final

# A step is only a candidate if this much of what we know about it is on the page. At 0.55
# a six-field step needs four of its labels and a three-field step needs two. Lower and a
# page sharing two common fields ("Full Name", "Email Address") with every other step
# starts winning; higher and a partly rendered page stops matching the step it is on.
MIN_STEP_SCORE: Final = 0.55

# How far ahead of the runner-up the best step must be. 0.62 against 0.60 is a coin toss,
# and a coin toss here fills the wrong page's fields. One clear field of difference on a
# six-field step is ~0.17, so 0.15 is just under "one field decided it".
MIN_STEP_MARGIN: Final = 0.15

# Added when a step's first two labels are both present. A partly scrolled or partly
# rendered page shows its opening fields first, so their presence is worth more than their
# share of the count suggests. Small enough that it cannot carry a step over
# MIN_STEP_SCORE on its own.
FIRST_LABELS_BONUS: Final = 0.1
FIRST_LABELS_COUNT: Final = 2

# Scores are rounded to this many places before any comparison. Without it, coverage
# arithmetic lands a hair under a threshold — 0.65 - 0.5 is 0.1499999999999999, which
# fails a >= 0.15 margin that a human reading the numbers would call a pass.
SCORE_PRECISION: Final = 4

# Longer than any real portal URL, and past the point where a pathological pattern could
# spend real time. An over-long URL is refused rather than truncated: a truncated URL can
# match a pattern the full one would not, which is the wrong direction to fail in.
MAX_URL_LENGTH: Final = 2048

# Rules change only when the seed runs, and `/context` loads a step's rules on every page
# read. Same window as the registry cache, for the same reason.
RULES_CACHE_TTL_SECONDS: Final = 300
RULES_CACHE_KEY_PREFIX: Final = "knowledge:rules:"
