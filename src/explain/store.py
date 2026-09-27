"""
The explanation cache: the key, the lookup, and the write.

Two properties hold this module together, and both are about what a cache must never do.

**It must never break an answer.** Every function here opens its own database session and
catches everything. A cache is an optimisation; a failed lookup is a miss and a failed write is
a slower second click. Sharing the request's transaction would mean a failed cache statement
poisoning it and turning a perfectly good explanation into a 500 at commit time — the one way a
cache can make a service less reliable than having none.

**It must never serve the wrong answer.** The key carries everything that changes what a correct
answer is: the workflow, the step, the normalised label, the prompt version, and the corpus's own
vintage. A re-ingest or a prompt change produces different keys, so yesterday's rows become
unreachable rather than stale — invalidation by construction, with no purge to remember. And a
custom question has **no key at all** (`cache_key_for` returns None), which is both why one
citizen's answer to their own question cannot be served to another and why their words never reach
a table every account can read.
"""

import hashlib
import logging
from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import async_session_factory
from src.explain.constants import (
    CACHE_KEY_SEPARATOR,
    CACHE_TTL_SECONDS,
    UNKNOWN_SCOPE_PART,
)
from src.explain.models import ExplanationCache
from src.explain.schemas import ExplainScope, SourceOut
from src.knowledge.normalize import normalize_label

logger = logging.getLogger(__name__)


def cache_key_for(
    scope: ExplainScope,
    label: str,
    *,
    prompt_version: str,
    corpus_stamp: str,
    has_question: bool,
) -> str | None:
    """
    The key for this field's explanation, or None when there must not be one.

    None in two cases, and both mean "there is no key that could safely match this answer later".

    A **custom question**, because the answer depends on words this service does not store. A flag
    in the key would have been the other way to write this, and it would have left the answer
    cacheable — which the spec says it is not, for the good reason that a question about one
    citizen's own application has no business in a row every other account can read.

    An **unlabelled field**, because the label is the only part of the key that distinguishes one
    field from another on the same step. Two unlabelled controls would otherwise share a row, and
    the second one asked about would be answered with an explanation of the first.

    The label is normalised with B8's `normalize_label`, so "Phone Number *" and "phone number"
    share a row — the same comparison the step matcher makes, rather than a second opinion about
    what two labels being the same means.

    Hashed, because one of the parts is a page label and a label on a government form can name a
    medical condition or a benefit. A key is for matching, and a hash matches just as well.
    """
    normalized = normalize_label(label)

    if has_question or not normalized:
        return None

    parts = (
        scope.workflow_id or UNKNOWN_SCOPE_PART,
        scope.step_id or UNKNOWN_SCOPE_PART,
        normalized,
        prompt_version,
        corpus_stamp,
    )
    joined = CACHE_KEY_SEPARATOR.join(parts)

    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


async def get_cached(key: str) -> ExplanationCache | None:
    """
    The stored answer for this key, or None.

    Filters `expires_at` in the query rather than after it, so a row past its week can never be
    served even if the cleanup sweep has not run. A database that does not answer is a miss: the
    caller pays for a model call, which is the right way to be wrong about a cache.
    """
    statement = select(ExplanationCache).where(
        ExplanationCache.cache_key == key,
        ExplanationCache.expires_at > func.now(),
    )

    try:
        async with async_session_factory() as db:
            result = await db.execute(statement)

            return result.scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001  # a miss is always an acceptable answer
        # Type only. The message can carry SQL, and a connection string with it.
        logger.warning("Explanation cache lookup failed (%s)", type(exc).__name__)

        return None


async def put_cached(
    key: str,
    *,
    explanation: str,
    example: str | None,
    sources: Sequence[SourceOut],
    prompt_version: str,
) -> bool:
    """
    Store one grounded answer. Returns whether it was written.

    An upsert on `cache_key`, so two panels asking about the same field at the same moment leave
    one row and the later answer wins — they are answers to the same question, and a conflict here
    is ordinary rather than exceptional.

    The caller decides what may be stored; this only stores it. What that means in practice is in
    `service.py`: an ungrounded answer and a custom question never get here.
    """
    expires_at = func.now() + timedelta(seconds=CACHE_TTL_SECONDS)
    values = {
        "cache_key": key,
        "explanation": explanation,
        "example": example,
        "sources": [source.model_dump() for source in sources],
        "prompt_version": prompt_version,
        "expires_at": expires_at,
    }

    statement = (
        insert(ExplanationCache)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[ExplanationCache.cache_key],
            set_={
                "explanation": values["explanation"],
                "example": values["example"],
                "sources": values["sources"],
                "prompt_version": values["prompt_version"],
                "expires_at": expires_at,
            },
        )
    )

    try:
        async with async_session_factory() as db:
            await db.execute(statement)
            await db.commit()
    except Exception as exc:  # noqa: BLE001  # a failed write costs a slower second click
        logger.warning("Explanation cache write failed (%s)", type(exc).__name__)

        return False

    return True


async def delete_expired_explanations(db: AsyncSession) -> int:
    """
    Delete every cached explanation past its expiry. Returns how many rows went.

    Takes a session, unlike the two functions above: this one is called by the cleanup CLI, which
    owns its own unpooled engine and sweeps the sessions table in the same transaction. The
    request path is where a cache must not touch its caller's transaction; a command whose entire
    job is deleting rows is not that path.
    """
    result = await db.execute(
        delete(ExplanationCache).where(ExplanationCache.expires_at <= func.now()),
    )

    return result.rowcount
