"""
The embedding client. The only module in the API that imports the OpenAI SDK.

Two rules hold here, and both exist because the prototype broke them:

1. **A missing key degrades, it does not crash.** The prototype raised at import, so one
   absent environment variable took the whole API down. Here `is_configured()` is false,
   every search reports unavailable, and `/health` says which.
2. **Nothing is logged but sizes.** What a citizen types into a government portal's help
   panel is sensitive by default, so no query text and no chunk text reaches a log line.
"""

import asyncio
import hashlib
import logging

from openai import AsyncOpenAI

from src.cache import cached
from src.config import settings
from src.documents.constants import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_TIMEOUT_SECONDS,
    MAX_QUERY_CHARS,
    QUERY_EMBEDDING_CACHE_MAX_ENTRIES,
    QUERY_EMBEDDING_CACHE_PREFIX,
    QUERY_EMBEDDING_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


class EmbeddingUnavailableError(RuntimeError):
    """The embedding could not be produced.

    Caught at the service boundary and turned into `available=False`; never raised into a
    request path.
    """


def is_configured() -> bool:
    """Whether an API key is set. False means retrieval answers `available=False`."""
    return bool(settings.openai_api_key)


def _get_client() -> AsyncOpenAI:
    """The shared async client, built on first use.

    Built lazily rather than at import so that a process with no key still imports this
    module — `/health` has to be able to report retrieval as unconfigured, which it cannot
    do from a module that refused to load.
    """
    global _client  # noqa: PLW0603  # one client per process, reused across requests

    if _client is None:
        if not is_configured():
            message = "OPENAI_API_KEY is not set"
            raise EmbeddingUnavailableError(message)

        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
            max_retries=1,
        )

    return _client


def _prepare(text: str) -> str:
    """Collapse newlines and cap the length before embedding.

    Newlines are collapsed because the prototype did and the corpus was embedded that way;
    a query shaped differently from the corpus measures further from it than it should.
    Over-long input is truncated rather than refused — the opening of a long question still
    carries what it is asking.
    """
    return text.replace("\n", " ").strip()[:MAX_QUERY_CHARS]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch, in order.

    One request for many inputs. The prototype embedded one chunk per call, which for a
    300-chunk corpus is 300 round trips for work the API takes in a handful.

    Raises `EmbeddingUnavailableError` on anything that goes wrong, including a missing key.
    """
    if not texts:
        return []

    client = _get_client()

    try:
        response = await client.embeddings.create(
            input=[_prepare(text) for text in texts],
            model=settings.embedding_model,
        )
    except Exception as exc:
        # The exception text can carry request bodies, which for us is user text. Only the
        # type is logged, and only the type is re-raised.
        logger.warning(
            "Embedding call failed (%s, inputs=%d)",
            type(exc).__name__,
            len(texts),
        )
        message = f"embedding call failed: {type(exc).__name__}"
        raise EmbeddingUnavailableError(message) from exc

    vectors = [item.embedding for item in response.data]

    if len(vectors) != len(texts):
        # Defensive: the caller pairs these with chunks positionally, and a short list
        # would attach the wrong vector to the wrong text rather than fail.
        message = f"expected {len(texts)} embeddings, got {len(vectors)}"
        raise EmbeddingUnavailableError(message)

    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            # A model of another width cannot go into a `vector(1536)` column, and finding
            # that out at INSERT time is a worse error message than this one.
            message = (
                f"{settings.embedding_model} returned {len(vector)} dimensions, "
                f"but the documents table stores {EMBEDDING_DIMENSIONS}"
            )
            raise EmbeddingUnavailableError(message)

    return vectors


def query_cache_key(query: str) -> str:
    """The cache key for one query: the model, plus a hash of the prepared text.

    Hashed rather than stored plainly because the key lives in a module-level dict for the
    life of the process, and the raw text is a question someone asked a government portal.
    The model is part of the key so changing it cannot serve a vector from the old one.
    """
    digest = hashlib.sha256(_prepare(query).encode("utf-8")).hexdigest()

    return f"{QUERY_EMBEDDING_CACHE_PREFIX}{settings.embedding_model}:{digest}"


async def embed_query(query: str) -> tuple[list[float], bool]:
    """One query's embedding, and whether it came from the cache.

    Repeat questions are common — the same field, the same fee, asked again on the next
    page load — and each embedding is a round trip on a path a user is watching.
    """
    key = query_cache_key(query)
    was_cached = True

    async def load() -> list[float]:
        nonlocal was_cached
        was_cached = False
        vectors = await embed_texts([query])

        return vectors[0]

    try:
        async with asyncio.timeout(EMBEDDING_TIMEOUT_SECONDS):
            vector = await cached(
                key,
                QUERY_EMBEDDING_CACHE_TTL_SECONDS,
                load,
                max_entries=QUERY_EMBEDDING_CACHE_MAX_ENTRIES,
            )
    except TimeoutError as exc:
        # The SDK has its own timeout; this bounds the whole await, retry included.
        logger.warning("Embedding timed out after %.1fs", EMBEDDING_TIMEOUT_SECONDS)
        message = "embedding timed out"
        raise EmbeddingUnavailableError(message) from exc

    return vector, was_cached


def reset_client() -> None:
    """Drop the cached client. For a script that changes settings after import."""
    global _client  # noqa: PLW0603

    _client = None
