"""
Load the prompt files and read their versions.

Prompts live as files in `prompts/` so a wording change reads as a wording change in a diff,
not as a Python edit. Each file starts with a version line — `<!-- v1 -->` — which is stripped
from the text the model sees and surfaced as a version instead. `PROMPT_VERSION` is logged with
every call, so when answer quality shifts you can tell which prompt produced it.

Reading four small local files at import is deterministic and has no database, HTTP or clock in
it, so it does not compromise the renderer's purity. A missing file or a missing version line
fails here, at import, rather than at the first model call.
"""

import re
from pathlib import Path
from typing import Final

_PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
_VERSION_LINE: Final = re.compile(r"^\s*<!--\s*(v\d+)\s*-->\s*\n?")


def _load(name: str) -> tuple[str, str]:
    """Return `(version, body)` for a prompt file, or fail loudly if it is malformed."""
    text = (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    match = _VERSION_LINE.match(text)

    if match is None:
        raise ValueError(f"prompt '{name}' is missing its leading version line, e.g. <!-- v1 -->")

    body = text[match.end() :].strip()

    return match.group(1), body


_SYSTEM_PLAN_VERSION, SYSTEM_PLAN = _load("system_plan")
_SYSTEM_EXPLAIN_VERSION, SYSTEM_EXPLAIN = _load("system_explain")
_USER_PLAN_VERSION, USER_PLAN = _load("user_plan")
_USER_EXPLAIN_VERSION, USER_EXPLAIN = _load("user_explain")

# The version of each prompt, for logging and debugging a quality regression.
PROMPT_VERSIONS: Final[dict[str, str]] = {
    "system_plan": _SYSTEM_PLAN_VERSION,
    "system_explain": _SYSTEM_EXPLAIN_VERSION,
    "user_plan": _USER_PLAN_VERSION,
    "user_explain": _USER_EXPLAIN_VERSION,
}


def _paired_version(first: str, second: str, first_version: str, second_version: str) -> str:
    """The shared version of a system/user prompt pair, or a failure at import."""
    if first_version != second_version:
        message = (
            f"{first} and {second} must share a version "
            f"(got {first_version} and {second_version})"
        )
        raise ValueError(message)

    return first_version


# The single version logged on a `/plan` call. A system prompt and its user template are one pair:
# they are versioned together, and a mismatch is a mistake worth catching at import.
PROMPT_VERSION: Final[str] = _paired_version(
    "system_plan",
    "user_plan",
    _SYSTEM_PLAN_VERSION,
    _USER_PLAN_VERSION,
)

# The version logged on an `/explain` call — and part of the explanation cache key, which is the
# other reason it has to be a single value. Bumping either explain prompt retires every cached
# answer by construction, rather than needing a purge someone has to remember to run.
EXPLAIN_PROMPT_VERSION: Final[str] = _paired_version(
    "system_explain",
    "user_explain",
    _SYSTEM_EXPLAIN_VERSION,
    _USER_EXPLAIN_VERSION,
)
