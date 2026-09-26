"""
Turning a page label into something comparable.

Conservative on purpose. Case, punctuation, required markers and whitespace go; nothing
else does. Stemming, stop-word removal and synonym expansion each buy a small gain and pay
for it with a class of wrong matches — "Business Address" matching "Business Name" is the
kind of mistake that puts a value in the wrong field of a government form.

Anything cleverer than an exact-match alias belongs in the model call, where a human sees
the result before it is used, not in a function whose output is filled in automatically.
"""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ALIASES_FILE = Path(__file__).parent / "data" / "label_aliases.json"

# Trailing markers a portal puts on a label without changing what the field is.
# `(required)`, `(optional)`, `*`, `:` and any run of them, in any order.
_TRAILING_MARKERS = re.compile(
    r"(?:\s*(?:\(\s*(?:required|optional)\s*\)|[*:]))+\s*$",
    re.IGNORECASE,
)

# Separators that stand in for a space. The last two are a non-breaking space and a
# narrow no-break space, which a portal's markup produces far more often than anyone expects.
_SEPARATORS = re.compile(r"[_\-/  ]+")

# Whatever is left that is not a letter, digit or space. Unicode-aware, so a name in a
# non-Latin script survives.
_NOT_ALNUM_OR_SPACE = re.compile(r"[^\w\s]|_", re.UNICODE)

_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    """
    The alias map, read once.

    Data rather than code so a teammate can add "gsm number" without opening an editor on a
    module. Both sides of the map are normalised on load, so a typo in the file's casing or
    punctuation cannot produce an alias that never fires.
    """
    if not ALIASES_FILE.exists():  # pragma: no cover - the file ships with the package
        return {}

    raw: dict[str, str] = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))

    return {_clean(key): _clean(value) for key, value in raw.items()}


def _clean(raw: str) -> str:
    """Everything `normalize_label` does except applying an alias."""
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = _TRAILING_MARKERS.sub("", text)
    text = _SEPARATORS.sub(" ", text)
    text = _NOT_ALNUM_OR_SPACE.sub("", text)

    return _WHITESPACE.sub(" ", text).strip()


def normalize_label(raw: str) -> str:
    """
    A page label reduced to its comparable form.

    `"Phone Number *"` gives `phone number`; `"E-mail Address:"` gives `email address`,
    via the alias map. Aliases are exact-match and applied last, so they can only rewrite a
    whole label and never a word inside one.

    Both sides of a comparison go through this — the registry's stored labels and the
    labels read off the page — so a seeded label and a page label that differ only by an
    alias still meet.
    """
    cleaned = _clean(raw)

    return _aliases().get(cleaned, cleaned)
