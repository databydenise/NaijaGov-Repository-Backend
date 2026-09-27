"""
Pure helpers for `/explain`: what kind of field this is, what to search for, and what to answer.

No database, no clock, no model, no randomness. Everything here is a function of the request, the
retrieved chunks and the turn's result — which is what lets the whole answer shape be checked
without a model call, and what keeps `service.py` a readable sequence rather than a wall.

The one judgement in this module is `sensitive_kind`, and it is the endpoint's most important
decision: it is what decides whether a model is called at all.
"""

from collections.abc import Sequence

from src.agent.schemas import ExplainTurn
from src.ai.schemas import Citation
from src.ai.utils import clean
from src.context.schemas import PageField
from src.documents.constants import MAX_QUERY_CHARS
from src.documents.schemas import RetrievedChunk
from src.explain.constants import (
    CANNED_EXPLANATIONS,
    GENERIC_SENSITIVE_EXPLANATION,
    SENSITIVE_KEYWORDS,
)
from src.explain.models import ExplanationCache
from src.explain.schemas import ExplainResponse, SourceOut

# The kind a sensitive field falls back to when its label matches none of the keyword lists.
GENERIC_KIND = "generic"

# The control type that is sensitive whatever its label says.
_PASSWORD_TYPE = "password"


def is_sensitive(page_field: PageField) -> bool:
    """
    Whether this field is one the Copilot never touches.

    Two signals, honoured together rather than one instead of the other: the content script's
    own `sensitive` flag, and a `type="password"` control. The same pair `/plan` uses to decide
    what to block, for the same reason — a page can say it either way, and missing one here would
    mean generating advice about a password box.
    """
    return page_field.sensitive or page_field.type.casefold() == _PASSWORD_TYPE


def sensitive_kind(page_field: PageField) -> str | None:
    """
    Which canned answer this field gets, or None when it is an ordinary field.

    The request carries no "kind" — only `sensitive: bool` — so it is read off the label and the
    type. The lists are walked in `SENSITIVE_KEYWORDS`' own order, most specific first: a
    `type="password"` box labelled "One-Time Code" is an OTP box, and telling its user to pick
    something they have not used elsewhere would be nonsense.

    A sensitive field whose label matches nothing gets `GENERIC_KIND`, whose copy claims nothing
    about what the field is. Guessing is what this whole endpoint is built to avoid.
    """
    if not is_sensitive(page_field):
        return None

    haystack = f"{page_field.label} {page_field.type}".casefold()

    for kind, keywords in SENSITIVE_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return kind

    return GENERIC_KIND


def canned_explanation(kind: str) -> str:
    """The fixed copy for a sensitive field's kind. Never formatted, never interpolated."""
    return CANNED_EXPLANATIONS.get(kind, GENERIC_SENSITIVE_EXPLANATION)


def build_query(page_field: PageField, nearby_text: str = "", question: str | None = None) -> str:
    """
    What to search the corpus for.

    A question rather than keywords, because that is what embeds well — the search tool's own
    description says as much, and the corpus is chunks of prose from agency FAQs. The portal's
    nearby text and the user's own question are appended when present: both name the thing the
    label abbreviates, and a label on its own ("Class") retrieves almost nothing.

    Everything is cleaned, because all three parts are untrusted text on their way to an embedding
    call, and truncated rather than refused — the opening of a long question still carries what it
    is asking.
    """
    subject = clean(page_field.label) or clean(page_field.type) or "this field"
    asked = f'What does the "{subject}" field on this form ask for, and what is required for it?'
    parts = [asked]

    if nearby_text:
        parts.append(clean(nearby_text))

    if question:
        parts.append(clean(question))

    return clean(" ".join(parts), MAX_QUERY_CHARS)


def to_sources(
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
) -> list[SourceOut]:
    """
    Verified citations, as the panel's source lines.

    `checked` is the chunk's own ingest date — when the corpus last read that page — never
    today's. The citations have already been rebuilt from the retrieved chunks by the time they
    get here, so a lookup that misses cannot happen; it is still handled, because a source with a
    URL and no title is worth more to a person than a 500 on the response path.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    sources: list[SourceOut] = []

    for citation in citations:
        chunk = by_id.get(citation.chunk_id)
        ingested_at = chunk.ingested_at if chunk else None

        sources.append(
            SourceOut(
                title=chunk.title if chunk else citation.source_url,
                url=citation.source_url,
                checked=ingested_at.date().isoformat() if ingested_at else None,
            ),
        )

    return sources


def fixed_answer(field_id: str, explanation: str, *, grounded: bool) -> ExplainResponse:
    """
    An answer that cost nothing: canned copy, or the no-guidance sentence.

    `grounded` is the caller's, because the two cases differ on exactly that. A sensitive field's
    answer is grounded — it is a fact about this product, with no official document to cite. A
    retrieval miss is not, and the panel shows it with the caveat that earns.
    """
    return ExplainResponse(
        field_id=field_id,
        explanation=explanation,
        example=None,
        sources=[],
        grounded=grounded,
        cached=False,
    )


def cached_answer(field_id: str, row: ExplanationCache) -> ExplainResponse:
    """
    A stored answer, for this field's id.

    The sources come from the row rather than from a fresh lookup: the chunk a stored answer cited
    may have been re-ingested under a new id since, and a source line rebuilt from a chunk that no
    longer exists would either vanish or come back wrong. Only grounded answers are stored, so
    `grounded` is true by construction.
    """
    return ExplainResponse(
        field_id=field_id,
        explanation=row.explanation,
        example=row.example,
        sources=[SourceOut.model_validate(source) for source in row.sources],
        grounded=True,
        cached=True,
    )


def answer_from_turn(field_id: str, outcome: ExplainTurn) -> ExplainResponse:
    """
    The model's answer, as the panel renders it.

    `grounded` is false only for `unverified`, which by this point means the reply has already been
    replaced with the guard's fixed sentence and its citations dropped. `not_required` — an
    explanation that makes no factual claim about the process — is grounded: there is nothing
    unsupported in it, and showing a caveat beside "this field asks for your surname" would teach
    the user to ignore caveats.
    """
    return ExplainResponse(
        field_id=field_id,
        explanation=outcome.response.explanation,
        example=outcome.response.example,
        sources=to_sources(outcome.response.citations, outcome.chunks),
        grounded=outcome.grounding != "unverified",
        cached=False,
    )
