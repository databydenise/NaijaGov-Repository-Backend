"""
Chunking, hashing, and text extraction for ingestion.

Pure functions, no I/O. They live under `src/` rather than in `scripts/` because the hash
is half of a database constraint: the value `scripts/ingest.py` computes has to be the same
value a later re-run computes, and a copy of this logic in a script is a copy that drifts.

Nothing here is imported by a request path.
"""

import hashlib
import re
from dataclasses import dataclass

# 1200 characters with 200 of overlap. Large enough that a requirement and its conditions
# stay in one chunk, small enough that three chunks are a reasonable amount of context.
# The overlap is what stops a sentence that straddles a boundary being retrievable from
# neither side.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

# Below this a chunk is a page header or a stray caption, not content worth an embedding.
MIN_CHUNK_CHARS = 60

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a source, with the title it will be shown under."""

    title: str
    content: str

    @property
    def content_sha256(self) -> str:
        return content_hash(self.content)


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and trim.

    Applied before hashing so that a page which re-indents its HTML, or a PDF extracted by
    a different pypdf version, does not present the same sentence as a new chunk.
    """
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(content: str) -> str:
    """SHA-256 of the whitespace-normalised chunk. Half of the dedupe key.

    Normalising inside the hash rather than at the call site means a caller cannot forget
    to, which would silently make every re-run insert duplicates again.
    """
    return hashlib.sha256(normalize_whitespace(content).encode("utf-8")).hexdigest()


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows, dropping anything too short to be content.

    Windows are cut at a character count rather than at sentences: the corpus is FAQ
    answers, portal copy, and PDF pages whose sentence boundaries survive extraction badly,
    and a fixed window with overlap is more predictable than a sentence splitter that meets
    "N50,000.00." halfway down a fee table.
    """
    cleaned = normalize_whitespace(text)

    if len(cleaned) <= size:
        return [cleaned] if len(cleaned) >= MIN_CHUNK_CHARS else []

    step = size - overlap
    chunks = []

    for start in range(0, len(cleaned), step):
        window = cleaned[start : start + size]

        if len(window) >= MIN_CHUNK_CHARS:
            chunks.append(window)

        # The last window reaches the end; anything further would repeat its tail.
        if start + size >= len(cleaned):
            break

    return chunks


def faq_chunks(question: str, answer: str, *, title_chars: int = 100) -> list[Chunk]:
    r"""One FAQ pair as `Question: …\nAnswer: …`, chunked if it runs long.

    The question is kept in the embedded text, not just the title: a user's phrasing
    matches a stored question far more closely than it matches the answer's wording.
    """
    question = normalize_whitespace(question)
    answer = normalize_whitespace(answer)

    if not question or not answer:
        return []

    body = f"Question: {question}\nAnswer: {answer}"
    title = f"FAQ: {question[:title_chars]}"
    parts = chunk_text(body)

    if len(parts) <= 1:
        return [Chunk(title=title, content=body)]

    return [
        Chunk(title=f"{title} (part {index})", content=part)
        for index, part in enumerate(parts, start=1)
    ]


def page_chunks(title: str, text: str) -> list[Chunk]:
    """A web page's extracted text, chunked and numbered."""
    parts = chunk_text(text)

    if len(parts) == 1:
        return [Chunk(title=title, content=parts[0])]

    return [
        Chunk(title=f"{title} (part {index})", content=part)
        for index, part in enumerate(parts, start=1)
    ]


def pdf_page_chunks(title: str, page_number: int, text: str) -> list[Chunk]:
    """One PDF page, chunked. The page number goes in the title so a citation can be found.

    A reader told "the 2025 Compendium says" cannot check it; told "page 47", they can.
    """
    parts = chunk_text(text)

    if len(parts) == 1:
        return [Chunk(title=f"{title} (page {page_number})", content=parts[0])]

    return [
        Chunk(
            title=f"{title} (page {page_number}, part {index})",
            content=part,
        )
        for index, part in enumerate(parts, start=1)
    ]
