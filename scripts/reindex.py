"""
Re-embed the corpus after a model change.

    python -m scripts.reindex --dry-run      # what would be re-embedded, and why
    python -m scripts.reindex                # do it, after confirming

A separate command from `ingest` because it costs real money: it pays for every row again,
where ingestion pays only for what is new. Nothing is deleted and nothing is re-fetched —
the text stays as it was scraped, and only `embedding` and `embedding_model` change.

Rows are selected by `embedding_model != EMBEDDING_MODEL`, so a run that stops partway can
simply be run again: what it already converted is no longer stale.
"""

import argparse
import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.session import create_cli_engine
from src.documents import embeddings
from src.documents.ingestion import (
    count_stale_embeddings,
    fetch_stale_batch,
    update_embedding,
)

# Same batching as ingestion, for the same reason.
BATCH_SIZE = 64

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count what would be re-embedded without spending anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt (for a non-interactive run)",
    )

    return parser.parse_args()


def _confirm(stale: int) -> bool:
    """Ask before spending.

    A re-embed is not reversible in the sense that matters: the money is gone whether or
    not the result was wanted.
    """
    print(
        f"\nThis will re-embed {stale} row(s) with {settings.embedding_model} "
        "and charge for every one of them.",
    )
    answer = input("Type the model name to continue: ").strip()

    return answer == settings.embedding_model


async def _reindex_batches(db: AsyncSession) -> tuple[int, int]:
    """Walk the stale rows by ascending id, re-embedding each batch. Returns (done, failed).

    Keyset pagination, not OFFSET: each batch stops being stale as soon as it is written,
    so the result set shrinks under the query and an offset would step over rows.
    """
    done = 0
    failed = 0
    after_id = 0

    while True:
        rows = await fetch_stale_batch(
            db,
            settings.embedding_model,
            after_id=after_id,
            batch_size=BATCH_SIZE,
        )

        if not rows:
            return done, failed

        after_id = rows[-1].id

        try:
            vectors = await embeddings.embed_texts([row.content for row in rows])
        except embeddings.EmbeddingUnavailableError as exc:
            print(f"  ! batch of {len(rows)} failed: {exc}")
            failed += len(rows)

            continue

        for row, vector in zip(rows, vectors, strict=True):
            await update_embedding(db, row.id, vector, settings.embedding_model)

        await db.commit()
        done += len(rows)
        print(f"  re-embedded {done} row(s)")


async def run(*, dry_run: bool, assume_yes: bool) -> int:
    """Returns a process exit code."""
    if not dry_run and not embeddings.is_configured():
        print("OPENAI_API_KEY is not set, so nothing can be embedded.", file=sys.stderr)

        return 1

    engine = create_cli_engine()

    try:
        async with AsyncSession(engine) as db:
            stale = await count_stale_embeddings(db, settings.embedding_model)

            if not stale:
                print(f"Every row is already embedded with {settings.embedding_model}.")

                return 0

            print(f"{stale} row(s) were embedded by a different model.")

            if dry_run:
                print("Dry run: nothing was changed.")

                return 0

            if not assume_yes and not _confirm(stale):
                print("Cancelled.")

                return 1

            done, failed = await _reindex_batches(db)
    finally:
        await engine.dispose()

    print(f"\nre-embedded {done} · {failed} failed")

    return 1 if failed else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args()

    sys.exit(asyncio.run(run(dry_run=args.dry_run, assume_yes=args.yes)))


if __name__ == "__main__":
    main()
