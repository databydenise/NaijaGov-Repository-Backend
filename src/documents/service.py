"""
Vector search over ingested official sources.

The one thing this module exists to do is **return nothing when nothing matches.**

In the prototype, `ORDER BY distance LIMIT 3` always returned three rows. A question about
renewal fees retrieved three chunks about renewal timing, and the model answered with a fee
that appeared in none of them and contradicted the FAQ that had been ingested. The cutoff
in the `WHERE` clause below is the fix, and every other line here is in service of it.

Nothing raises into a request path. A search that cannot be made comes back as
`RetrievalResult([], available=False)`, which a caller must not confuse with an empty
corpus answer.
"""

import logging
import time
from datetime import datetime
from typing import Any

from sqlalchemy import Row, Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache import cached
from src.config import settings
from src.database.session import async_session_factory
from src.documents.constants import (
    DEFAULT_SEARCH_LIMIT,
    LATEST_INGEST_CACHE_PREFIX,
    LATEST_INGEST_CACHE_TTL_SECONDS,
    MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT,
    SEARCH_STATEMENT_TIMEOUT_MS,
)
from src.documents.embeddings import (
    EmbeddingUnavailableError,
    embed_query,
    is_configured,
)
from src.documents.models import Document
from src.documents.schemas import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

def _empty() -> RetrievalResult:
    """Nothing found, and the search did run. The caller says "no official guidance".

    A function rather than a module-level constant: `RetrievalResult` is frozen, but the
    list inside it is not, and a shared instance would let one caller's `.sort()` or
    `.append()` reach every later search.
    """
    return RetrievalResult(chunks=[], available=True)


def _unavailable() -> RetrievalResult:
    """The search could not be made. Not evidence of absence."""
    return RetrievalResult(chunks=[], available=False)


def _clamp_limit(limit: int) -> int:
    """Keep `limit` inside 1–5, whatever a caller or a model asked for."""
    return max(MIN_SEARCH_LIMIT, min(limit, MAX_SEARCH_LIMIT))


def _build_statement(
    embedding: list[float],
    *,
    limit: int,
    max_distance: float,
    agency: str | None,
    service: str | None,
) -> Select[Any]:
    """The search, with the cutoff in the WHERE clause rather than applied afterwards.

    Filtering in SQL rather than in Python is not a micro-optimisation: `LIMIT` applies
    after `WHERE`, so a cutoff in the query returns the closest *qualifying* rows, while a
    cutoff applied to an already-limited result silently throws away matches that were
    behind three near-misses.
    """
    distance = Document.embedding.cosine_distance(embedding).label("distance")

    statement = (
        select(
            Document.id,
            Document.title,
            Document.content,
            Document.source_url,
            Document.agency,
            Document.service,
            Document.ingested_at,
            distance,
        )
        .where(distance < max_distance)
        .order_by(distance)
        .limit(limit)
    )

    # Exact matches on the stored strings. The caller passes what the registry holds, so
    # there is nothing to normalise; a typo returning nothing is the right failure.
    if agency:
        statement = statement.where(Document.agency == agency)
    if service:
        statement = statement.where(Document.service == service)

    return statement


def _to_chunk(row: Row[Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        title=row.title,
        content=row.content,
        source_url=row.source_url,
        agency=row.agency,
        service=row.service,
        distance=float(row.distance),
        ingested_at=row.ingested_at,
    )


async def search_government_information(
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    max_distance: float | None = None,
    agency: str | None = None,
    service: str | None = None,
) -> RetrievalResult:
    """
    Search the ingested corpus, and return the chunks that are actually close enough.

    An empty `chunks` list with `available=True` is a real answer and the expected outcome
    for a question the corpus does not cover. The caller must relay that rather than let a
    model fill the gap.

    Never raises, and — unlike every other service function here — takes no session. It
    opens its own from the shared pool, for two reasons that both bite on the `/plan` path:

    - a statement timeout aborts the transaction it fires in, so sharing the caller's
      transaction would mean a slow search discarding a session row the caller had already
      written;
    - `SET LOCAL` lasts to the end of its transaction, so the two-second budget would
      silently apply to every later statement the caller ran.

    A read that must never affect its caller does not belong in its caller's transaction.
    """
    query = query.strip()

    if not query:
        return _empty()

    if not is_configured():
        # Nothing was searched. Distinct from "nothing matched", and `/health` says so.
        logger.warning("Retrieval asked for, but OPENAI_API_KEY is not set")

        return _unavailable()

    limit = _clamp_limit(limit)
    cutoff = settings.retrieval_max_distance if max_distance is None else max_distance
    started = time.perf_counter()

    try:
        embedding, cache_hit = await embed_query(query)
    except EmbeddingUnavailableError:
        # Already logged, by type only, inside the embedding client.
        return _unavailable()

    try:
        async with async_session_factory() as db:
            # Postgres-side, so a slow scan releases the connection on its own rather than
            # being abandoned client-side while it keeps running. Discarded with this
            # transaction when the session closes.
            await db.execute(
                text(f"SET LOCAL statement_timeout = {SEARCH_STATEMENT_TIMEOUT_MS}"),
            )

            result = await db.execute(
                _build_statement(
                    embedding,
                    limit=limit,
                    max_distance=cutoff,
                    agency=agency,
                    service=service,
                ),
            )
            rows = result.all()
    except Exception as exc:  # noqa: BLE001  # a request path must survive any db failure
        # Type only. The message can carry SQL, and a connection string with it.
        logger.warning(
            "Vector search failed (%s, query_length=%d)",
            type(exc).__name__,
            len(query),
        )

        return _unavailable()

    chunks = [_to_chunk(row) for row in rows]

    _log_search(
        query_length=len(query),
        chunks=chunks,
        cutoff=cutoff,
        cache_hit=cache_hit,
        duration_ms=(time.perf_counter() - started) * 1000,
    )

    return RetrievalResult(chunks=chunks, available=True)


def _log_search(
    *,
    query_length: int,
    chunks: list[RetrievedChunk],
    cutoff: float,
    cache_hit: bool,
    duration_ms: float,
) -> None:
    """One line per search: sizes, counts and timings only.

    Never the query text and never chunk content. What someone asks while filling in a
    government form is sensitive by default, and a log is the easiest place to leak it.
    `best_distance` is the number the cutoff was compared against, which is what a later
    sweep needs and what explains an empty result without re-running it.
    """
    logger.info(
        "retrieval query_length=%d results=%d best_distance=%s cutoff=%.2f "
        "cache_hit=%s duration_ms=%.0f",
        query_length,
        len(chunks),
        f"{chunks[0].distance:.4f}" if chunks else "none",
        cutoff,
        cache_hit,
        duration_ms,
    )


async def _load_latest_ingested_at(db: AsyncSession, agency: str | None) -> datetime | None:
    """`MAX(ingested_at)` over the corpus, or over one agency's part of it."""
    statement = select(func.max(Document.ingested_at))

    if agency:
        statement = statement.where(Document.agency == agency)

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def latest_ingested_at(db: AsyncSession, agency: str | None = None) -> datetime | None:
    """
    When this agency's material was last ingested, or None if it has none.

    The corpus's vintage, which is what makes cache invalidation by construction possible: a caller
    that puts this timestamp in a cache key gets fresh keys the moment new material is ingested,
    and the rows written against the old vintage are simply never looked up again. No purge, no
    delete path, and nothing to remember to run after an ingest.

    Never raises. A database that does not answer comes back as None, and a caller must treat that
    as "vintage unknown" rather than as "nothing ingested" — the two would otherwise share a cache
    key, so an answer written during an outage could be served afterwards.
    """
    try:
        return await cached(
            f"{LATEST_INGEST_CACHE_PREFIX}{agency or '*'}",
            LATEST_INGEST_CACHE_TTL_SECONDS,
            lambda: _load_latest_ingested_at(db, agency),
        )
    except Exception as exc:  # noqa: BLE001  # a request path must survive any db failure
        # Type only. The message can carry SQL, and a connection string with it.
        logger.warning("Could not read the corpus vintage (%s)", type(exc).__name__)

        return None

