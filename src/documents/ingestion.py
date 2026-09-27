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
    """Insert chunks, skipping any that collide on `(source_url, content_sha256)`.

    `ON CONFLICT DO NOTHING` on top of the hash check above, not instead of it. The hash
    check saves the embedding spend; this closes the window where two runs overlap, and
    makes the guarantee the schema's rather than the script's.

    Returns how many rows were actually written.
    """
    if not documents:
        return 0

    statement = (
        insert(Document)
        .values(
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
        .on_conflict_do_nothing(index_elements=["source_url", "content_sha256"])
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
