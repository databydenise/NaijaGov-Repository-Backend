"""
What retrieval hands back.

Evidence, never prose: the chunk, where it came from, and how close it was. `available`
is separate from an empty `chunks` list on purpose, because the two mean opposite things
to a caller — "no source covers this" is an answer to relay, and "the search did not run"
is not.

Frozen dataclasses rather than pydantic models: nothing here crosses an HTTP boundary or
parses untrusted input, and these are built on a hot path. `RuleOut` in `knowledge/` is a
pydantic model because it is serialised into a response; these are not.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk of official source material, with its provenance and its distance."""

    # The `documents.id` row. Carried through to the agent's tool result so the guard can
    # check that a citation refers to something that was actually retrieved.
    chunk_id: int

    title: str
    content: str
    source_url: str
    agency: str
    service: str

    # Raw cosine distance: 0 is identical, 2 is opposite. Kept rather than discarded
    # because tuning the cutoff needs the number the cutoff is compared against.
    distance: float

    # When this chunk was ingested — the corpus's own "last checked" date, shown beside
    # every citation that uses it. Optional only so a hand-built chunk in a check does not
    # have to invent one; the search always sets it. A citation shown without it says so,
    # rather than borrowing today's date and implying the source was read today.
    ingested_at: datetime | None = None

    # `1 - distance`, for a log line or a panel that wants "closer is bigger". Derived, not
    # stored, and never the thing a threshold is applied to.
    @property
    def score(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of one search.

    `chunks == [] and available` means the corpus genuinely has nothing on the question.
    The caller must say so rather than answer from the model's own knowledge.

    `not available` means the search could not be made — no API key, the embedding call
    failed, the pool was exhausted, the query timed out. The caller must not treat it as
    evidence of absence.
    """

    chunks: list[RetrievedChunk]
    available: bool

    @property
    def is_empty(self) -> bool:
        return not self.chunks


@dataclass(frozen=True)
class CorpusStats:
    """What `/health` reports about the ingested corpus."""

    documents: int
    distinct_sources: int
