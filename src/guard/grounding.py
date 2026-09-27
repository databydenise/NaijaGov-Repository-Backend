"""
Does the reply make a factual claim, and is that claim backed by something we retrieved?

This is the check that exists because of a specific failure: the prototype answered with a
confident fee that appeared in none of the retrieved text. A citizen cannot tell that sentence
from a true one, and neither can we without looking.

So the reply is scanned for *claim markers* — an amount, a percentage, a duration, a count of
documents, a normative "you must" — and a reply carrying one needs a citation that points at a
chunk retrieval actually returned this turn. The marker set is regex, not a model call, and it is
deliberately loose in the direction that is cheap to be wrong in: a false `unverified` costs one
extra model call, a false `grounded` costs a citizen wrong information about their own government.

Citations are rebuilt rather than trusted. The `chunk_id` must be among the ids retrieved, and the
`source_url` is taken from the chunk — never from the model, which has every opportunity to pair a
real-looking gov.ng URL with a claim that URL does not make.
"""

import re
from collections.abc import Sequence
from typing import Final

from src.ai.schemas import Citation
from src.documents.schemas import RetrievedChunk
from src.guard.constants import GroundingVerdict

# Counts written as words, for the patterns below. Beyond ten a reply almost always uses digits.
_COUNT_WORD: Final = r"(?:one|two|three|four|five|six|seven|eight|nine|ten)"

_CLAIM_MARKERS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    # An amount of money: ₦5,000 · NGN 5000 · N5,000 · "five thousand naira". The bare `N`
    # form only counts when a digit follows, so an ordinary sentence starting with N is safe.
    (
        "currency",
        re.compile(r"(?:₦\s?[\d,]+|\bNGN\s?[\d,]+|\bN\s?[\d][\d,]*|\bnaira\b)", re.IGNORECASE),
    ),
    # A percentage, in digits or words. Fees on Nigerian portals are often quoted as one.
    (
        "percentage",
        re.compile(r"(?:\d+(?:\.\d+)?\s*%|\b\d+\s*per\s?cent\b|\bpercent\b)", re.IGNORECASE),
    ),
    # A duration: "14 working days", "3 years", "two weeks". The optional qualifier matters —
    # "working days" and "days" are different promises to someone planning a trip to an office.
    (
        "duration",
        re.compile(
            rf"\b(?:\d+|{_COUNT_WORD})\s*(?:working\s+|business\s+|calendar\s+)?"
            r"(?:minute|hour|day|week|month|year)s?\b",
            re.IGNORECASE,
        ),
    ),
    # A count of what to bring: "two passport photographs", "3 copies", "one certificate".
    (
        "document_count",
        re.compile(
            rf"\b(?:\d+|{_COUNT_WORD})\s+(?:passport\s+)?"
            r"(?:document|copy|copies|photograph|photo|form|certificate|id)s?\b",
            re.IGNORECASE,
        ),
    ),
    # A rule stated as a rule. "You must", "is required", "mandatory" — the phrasing that turns
    # a guess into an instruction the user will act on.
    (
        "normative",
        re.compile(
            r"\b(?:you\s+(?:must|need\s+to|have\s+to|are\s+required\s+to|cannot|can't)"
            r"|(?:is|are)\s+required|require[sd]?\b|mandatory|not\s+eligible|eligible\s+for)",
            re.IGNORECASE,
        ),
    ),
)


def claim_markers(reply: str) -> tuple[str, ...]:
    """The names of the markers this reply trips. Names only — never the matched text."""
    return tuple(name for name, pattern in _CLAIM_MARKERS if pattern.search(reply))


def verify_citations(
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[Citation], int]:
    """
    The citations that survive, rebuilt from the chunks, and how many were dropped.

    A citation is kept only if its `chunk_id` is one retrieval returned this turn, and the kept
    citation carries the chunk's own `source_url`. Repeats collapse, so three citations to one
    chunk do not read as three sources. A citation the model invented is worse than none: it
    makes an unsupported claim look checked.
    """
    retrieved = {chunk.chunk_id: chunk for chunk in chunks}
    verified: list[Citation] = []
    seen: set[int] = set()

    for citation in citations:
        chunk = retrieved.get(citation.chunk_id)
        if chunk is None or citation.chunk_id in seen:
            continue

        seen.add(citation.chunk_id)
        verified.append(Citation(chunk_id=chunk.chunk_id, source_url=chunk.source_url))

    return verified, len(citations) - len(verified)


def grounding_verdict(reply: str, verified_citations: Sequence[Citation]) -> GroundingVerdict:
    """
    `not_required` when the reply claims nothing, `grounded` when a claim has a verified
    citation, `unverified` otherwise.

    Note what this does *not* check: that the cited chunk actually says the thing claimed. That
    needs a second model call, and a citation to a retrieved chunk is the strongest guarantee
    this fragment can give without one.
    """
    if not claim_markers(reply):
        return "not_required"

    if verified_citations:
        return "grounded"

    return "unverified"
