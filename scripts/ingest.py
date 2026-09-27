"""
Fill the `documents` table from the sources in `data/sources.yaml`.

Run by hand, output reviewed, never from the API:

    python -m scripts.ingest                     # ingest everything new
    python -m scripts.ingest --dry-run           # what would be new, costing nothing
    python -m scripts.ingest --only frsc.gov.ng  # one source, by URL substring

A second run inserts nothing and spends nothing: each chunk is hashed, the hashes already
stored for that URL are read first, and only what is genuinely new is embedded. The unique
index on `(source_url, content_sha256)` is the backstop, so the guarantee is the schema's
and not this script's.
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from scripts.sources import (
    Source,
    SourceError,
    build_client,
    collect_chunks,
    load_manifest,
)
from src.config import settings
from src.database.session import create_cli_engine
from src.documents import embeddings
from src.documents.ingestion import PendingDocument, existing_hashes, insert_documents
from src.documents.utils import Chunk

DEFAULT_MANIFEST = "data/sources.yaml"

# Inputs per embedding request. At ~1200 characters a chunk this is roughly 20k tokens a
# call, well inside the limit, and it turns a 300-chunk corpus into five requests rather
# than the prototype's 300.
EMBED_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


@dataclass
class RunTotals:
    """What the closing summary line reports."""

    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    failed_sources: list[str] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="path to sources.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what is new without embedding or inserting anything",
    )
    parser.add_argument(
        "--only",
        help="ingest only sources whose URL contains this substring",
    )

    return parser.parse_args()


def _deduplicate(chunks: list[Chunk], stored: set[str]) -> tuple[list[Chunk], int]:
    """
    Split chunks into those worth embedding and a count of those already held.

    Chunks are also deduplicated against each other. Overlapping windows over a repetitive
    page can produce the same text twice in one run, which would otherwise be one insert
    and one silent `ON CONFLICT` — the same bug, just caught later and after paying for the
    embedding.
    """
    fresh: list[Chunk] = []
    seen: set[str] = set()
    skipped = 0

    for chunk in chunks:
        digest = chunk.content_sha256

        if digest in stored or digest in seen:
            skipped += 1

            continue

        seen.add(digest)
        fresh.append(chunk)

    return fresh, skipped


async def _embed_and_store(
    db: AsyncSession,
    source: Source,
    chunks: list[Chunk],
) -> tuple[int, int]:
    """Embed the new chunks in batches and insert them. Returns (inserted, failed)."""
    inserted = 0
    failed = 0

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]

        try:
            vectors = await embeddings.embed_texts([chunk.content for chunk in batch])
        except embeddings.EmbeddingUnavailableError as exc:
            # One batch failing is not a reason to abandon the rest of the corpus, and the
            # ones that did land stay landed — the next run picks up where this one stopped.
            print(f"    ! batch of {len(batch)} failed to embed: {exc}")
            failed += len(batch)

            continue

        pending = [
            PendingDocument(
                title=chunk.title,
                agency=source.agency,
                service=source.service,
                content=chunk.content,
                content_sha256=chunk.content_sha256,
                source_url=source.url,
                source_kind=source.kind,
                embedding=vector,
                embedding_model=settings.embedding_model,
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        written = await insert_documents(db, pending)
        await db.commit()

        # A row the unique index rejected was stored by an earlier or concurrent run, so
        # `written` can be below `len(batch)`. That is not a failure, and counting it as
        # one would make a clean re-run look broken.
        inserted += written
        print(f"    embedded {len(batch)}, inserted {written}")

    return inserted, failed


async def _ingest_source(
    db: AsyncSession | None,
    client: httpx.Client,
    source: Source,
    *,
    dry_run: bool,
) -> RunTotals:
    """
    One source, start to finish. Never raises: a bad source is reported and skipped.

    `db` is None only on a dry run with no reachable database, where the point is to check
    that a source still extracts — the selectors on a government page change without
    notice, and finding that out should not require a database.
    """
    totals = RunTotals()
    print(f"\n{source.kind:>4}  {source.url}")

    try:
        chunks = collect_chunks(client, source)
    except SourceError as exc:
        print(f"    ! {exc}")
        totals.failed_sources.append(source.url)

        return totals

    stored = await existing_hashes(db, source.url) if db is not None else set()
    fresh, skipped = _deduplicate(chunks, stored)
    totals.skipped = skipped

    print(f"    {len(chunks)} chunks · {len(fresh)} new · {skipped} already stored")

    if not fresh:
        return totals

    if dry_run:
        print(f"    (dry run — would embed {len(fresh)})")

        return totals

    if db is None:
        message = "A database session is required to store ingested chunks"
        raise RuntimeError(message)

    totals.ingested, totals.failed = await _embed_and_store(db, source, fresh)

    return totals


async def _open_session(engine: AsyncEngine, *, dry_run: bool) -> AsyncSession | None:
    """
    A session, or None when a dry run has no database to talk to.

    A real run needs one and lets the failure through. A dry run does not: checking that a
    page still yields the chunks we expect is worth doing on a laptop with no Postgres, and
    it is the check most likely to be wanted in a hurry when a portal has been redesigned.
    """
    session = AsyncSession(engine)

    try:
        await session.execute(select(1))
    except Exception as exc:  # any driver error means there is no database
        await session.close()

        if not dry_run:
            raise

        print(
            f"  (no database: {type(exc).__name__}. Every chunk will be reported as new, "
            "because what is already stored cannot be read.)",
        )

        return None

    return session


async def run(manifest_path: str, *, dry_run: bool, only: str | None) -> int:
    """Ingest every source in the manifest. Returns a process exit code."""
    try:
        sources = load_manifest(manifest_path)
    except (OSError, SourceError) as exc:
        print(f"Could not read {manifest_path}: {exc}", file=sys.stderr)

        return 1

    if only:
        sources = [source for source in sources if only in source.url]

        if not sources:
            print(f"No source in {manifest_path} matches '{only}'", file=sys.stderr)

            return 1

    if not dry_run and not embeddings.is_configured():
        print(
            "OPENAI_API_KEY is not set, so nothing can be embedded.\n"
            "Set it in .env, or use --dry-run to see what would be ingested.",
            file=sys.stderr,
        )

        return 1

    print(f"Ingesting {len(sources)} source(s) with model {settings.embedding_model}")

    if dry_run:
        print("Dry run: nothing will be embedded, nothing will be written.")

    totals = RunTotals()
    engine = create_cli_engine()
    client = build_client()

    try:
        db = await _open_session(engine, dry_run=dry_run)

        try:
            for source in sources:
                result = await _ingest_source(db, client, source, dry_run=dry_run)
                totals.ingested += result.ingested
                totals.skipped += result.skipped
                totals.failed += result.failed
                totals.failed_sources.extend(result.failed_sources)
        finally:
            if db is not None:
                await db.close()
    finally:
        client.close()
        await engine.dispose()

    print(
        f"\ningested {totals.ingested} new · "
        f"skipped {totals.skipped} existing · "
        f"{totals.failed} failed",
    )

    if totals.failed_sources:
        print(f"sources that could not be read: {', '.join(totals.failed_sources)}")

    return 1 if totals.failed or totals.failed_sources else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args()

    sys.exit(asyncio.run(run(args.manifest, dry_run=args.dry_run, only=args.only)))


if __name__ == "__main__":
    main()
