"""
Validating what the model sent before anything runs.

Tool arguments are untrusted input in the most literal sense: the model wrote them, and a model
that has just read a hostile page wrote them under that influence. Every one is checked here
before a query reaches the database, and a failure is *answered* rather than raised — a missing
quote must not end a turn the user is watching.

The checks are deliberately narrow. This is not a place to be helpful: a limit outside 1–5 is
clamped, whitespace is trimmed, and everything else that does not fit is refused with a sentence
telling the model what to send instead. Guessing at what it meant is how a search for the wrong
thing comes back looking like evidence.
"""

import json
from dataclasses import dataclass
from typing import Any

from src.agent.constants import BAD_ARGUMENTS_MESSAGE
from src.documents.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_QUERY_CHARS,
    MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT,
)


@dataclass(frozen=True)
class SearchArgs:
    """Validated arguments for one search. Built only from arguments that passed every check."""

    query: str
    limit: int
    agency: str | None
    service: str | None

    @property
    def cache_key(self) -> tuple[str, int, str, str]:
        """
        What makes two searches the same search within a turn.

        The query is case-folded, because "Renewal fee?" and "renewal fee?" embed to the same
        place and asking twice costs a round trip for a result we already hold.
        """
        return (self.query.casefold(), self.limit, self.agency or "", self.service or "")


def _clamp_limit(value: int) -> int:
    """Keep `limit` inside 1–5, whatever the model asked for."""
    return max(MIN_SEARCH_LIMIT, min(value, MAX_SEARCH_LIMIT))


def _optional_string(value: Any) -> tuple[str | None, bool]:  # noqa: ANN401  # untrusted JSON
    """An optional filter: `(value, ok)`. Absent is fine; a non-string is not."""
    if value is None:
        return None, True

    if not isinstance(value, str):
        return None, False

    return (value.strip() or None), True


def parse_search_args(raw: str) -> SearchArgs | str:
    """
    Validate what the model sent, and return either usable arguments or the reason they are not.

    Returns a `str` on failure — the sentence the model is shown — rather than raising, because a
    malformed tool call is the model's mistake to correct inside its own turn.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return "The arguments were not valid JSON. " + BAD_ARGUMENTS_MESSAGE

    if not isinstance(parsed, dict):
        return "The arguments must be a JSON object. " + BAD_ARGUMENTS_MESSAGE

    query = parsed.get("query")

    if not isinstance(query, str) or not query.strip():
        return "'query' must be a non-empty string. " + BAD_ARGUMENTS_MESSAGE

    limit = parsed.get("limit", DEFAULT_SEARCH_LIMIT)

    # `bool` is an `int` in Python, and `limit: true` is not a count.
    if isinstance(limit, bool) or not isinstance(limit, int):
        return "'limit' must be a whole number. " + BAD_ARGUMENTS_MESSAGE

    agency, agency_ok = _optional_string(parsed.get("agency"))
    service, service_ok = _optional_string(parsed.get("service"))

    if not (agency_ok and service_ok):
        return "'agency' and 'service' must be strings. " + BAD_ARGUMENTS_MESSAGE

    return SearchArgs(
        query=query.strip()[:MAX_QUERY_CHARS],
        limit=_clamp_limit(limit),
        agency=agency,
        service=service,
    )
