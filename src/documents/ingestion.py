"""
Database writes for ingestion. Separate from `service.py`, which only reads.

Nothing here fetches anything. Every function takes text that a script already has, so no
module under `src/` can reach a government website — the scraping lives in `scripts/`,
which the API never imports. That split is the structural version of the project rule that
this service never fetches, scrapes, or submits to a government site at request time.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.documents.models import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingDocument:
    """A chunk ready to be stored, with its embedding already computed."""

    title: str
    agency: str
    service: str
    content: str
    content_sha256: str
    source_url: str
    source_kind: str
    embedding: list[float]
    embedding_model: str


async def existing_hashes(db: AsyncSession, source_url: str) -> set[str]:
    """Every chunk hash already stored for one source URL.

    Read before embedding, not after: the point of the dedupe is that a second run of
    ingestion costs no embedding calls, and a check that happens at INSERT time has
    already paid for them.
    """
    statement = select(Document.content_sha256).where(Document.source_url == source_url)
    result = await db.execute(statement)

    return set(result.scalars().all())


async def insert_documents(db: AsyncSession, documents: Sequence[PendingDocument]) -> int:
    """
    Insert chunks. Every one of them — there is no longer a constraint to skip on.

    Migration 0010 dropped `uq_documents_source_content`, on request, so a corpus already
    containing exact-content duplicates across several ingestion runs could be loaded whole
    rather than have the index silently keep one copy of each. That also removes the
    database-side backstop `scripts.ingest` used to lean on: its own `existing_hashes()`
    check, made before any embedding call, is now the only thing standing between a normal
    ingestion run and a duplicate row, and two overlapping runs are no longer caught by
    anything downstream of that check.

    Returns how many rows were written, which with no conflict target is always
    `len(documents)` — kept as a return value so a caller does not have to know that.
    """
    if not documents:
        return 0

    statement = insert(Document).values(
        [
            {
                "title": document.title,
                "agency": document.agency,
                "service": document.service,
                "content": document.content,
                "content_sha256": document.content_sha256,
                "source_url": document.source_url,
                "source_kind": document.source_kind,
                "embedding": document.embedding,
                "embedding_model": document.embedding_model,
            }
            for document in documents
        ],
    )

    result = await db.execute(statement)

    return result.rowcount


async def count_stale_embeddings(db: AsyncSession, current_model: str) -> int:
    """How many rows were embedded by some other model. What `reindex` is about to charge for."""
    statement = (
        select(func.count(Document.id))
        .where(Document.embedding_model != current_model)
    )

    return (await db.execute(statement)).scalar_one()


async def fetch_stale_batch(
    db: AsyncSession,
    current_model: str,
    *,
    after_id: int,
    batch_size: int,
) -> Sequence[Document]:
    """The next batch of rows embedded by an older model, by ascending id.

    Keyset pagination rather than OFFSET: each batch is updated as it is read, so the set
    of stale rows shrinks under the query and an offset would skip rows.
    """
    statement = (
        select(Document)
        .where(Document.embedding_model != current_model, Document.id > after_id)
        .order_by(Document.id)
        .limit(batch_size)
    )
    result = await db.execute(statement)

    return result.scalars().all()


async def update_embedding(
    db: AsyncSession,
    document_id: int,
    embedding: list[float],
    embedding_model: str,
) -> None:
    """Replace one row's vector and record which model produced it."""
    statement = (
        update(Document)
        .where(Document.id == document_id)
        .values(embedding=embedding, embedding_model=embedding_model)
    )

    await db.execute(statement)
