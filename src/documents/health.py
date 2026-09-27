"""
What `/health` says about retrieval.

Separate from `service.py` so that the search path imports nothing it does not need, and
so the reporting can answer even when the search cannot.

The distinction this module exists to publish: an instance with no API key, and an instance
with an empty corpus, are both *up*. Both will answer every question with "no official
guidance", and without this block they would look identical to a healthy one.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.cache import cached
from src.config import settings
from src.documents.constants import (
    CORPUS_STATS_CACHE_KEY,
    CORPUS_STATS_CACHE_TTL_SECONDS,
    RetrievalStatus,
)
from src.documents.embeddings import is_configured
from src.documents.models import Document
from src.documents.schemas import CorpusStats

logger = logging.getLogger(__name__)


async def get_corpus_stats(db: AsyncSession) -> CorpusStats:
    """How much material is ingested, and how many distinct pages it came from."""
    statement = select(
        func.count(Document.id),
        func.count(func.distinct(Document.source_url)),
    )
    documents, distinct_sources = (await db.execute(statement)).one()

    return CorpusStats(documents=documents, distinct_sources=distinct_sources)


async def get_cached_corpus_stats(db: AsyncSession) -> CorpusStats:
    """The corpus counts, cached for five minutes.

    `/health` is polled, and this counts a table only an ingestion run writes to.
    """

    async def load() -> CorpusStats:
        return await get_corpus_stats(db)

    return await cached(CORPUS_STATS_CACHE_KEY, CORPUS_STATS_CACHE_TTL_SECONDS, load)


async def retrieval_health(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, str | int | None]:
    """The `retrieval` block of `/health`.

    Never raises. `/health` is a liveness check first: an instance that is up says so even
    when the database is unreachable, and the counts degrade to null rather than taking the
    endpoint down with them.
    """
    if not is_configured():
        # No key. The app runs, and every search reports unavailable. Reported rather than
        # kept quiet, because the symptom otherwise looks like an empty corpus.
        return {
            "status": RetrievalStatus.UNCONFIGURED,
            "documents": None,
            "distinct_sources": None,
            "embedding_model": settings.embedding_model,
        }

    try:
        async with session_factory() as db:
            stats = await get_cached_corpus_stats(db)
    except Exception as exc:  # noqa: BLE001  # liveness must survive a database failure
        logger.warning("Could not read the corpus stats (%s)", type(exc).__name__)

        return {
            "status": RetrievalStatus.UNAVAILABLE,
            "documents": None,
            "distinct_sources": None,
            "embedding_model": settings.embedding_model,
        }

    return {
        "status": RetrievalStatus.OK,
        "documents": stats.documents,
        "distinct_sources": stats.distinct_sources,
        "embedding_model": settings.embedding_model,
    }
