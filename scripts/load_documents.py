"""
Load a polished corpus CSV into the `documents` table.

    python -m scripts.load_documents --dry-run    # validate the file, touch nothing
    python -m scripts.load_documents              # load data/documents_corpus.csv
    python -m scripts.load_documents --input other.csv

A script and not a migration, deliberately. A migration describes the shape of the schema
and runs once, forward, on every database including an empty test one; a corpus is content,
it is measured in megabytes of vectors, and it is replaced when a source is re-scraped.
Putting these rows in `alembic/versions/` would make every future `alembic upgrade head`
carry them and make replacing them a second migration. `documents` is filled by hand, by
`scripts.ingest`, and this is the same job from a file instead of a website.

Embeddings come from the CSV, so no OpenAI key is needed and nothing is charged. That is the
reason this path exists: the vectors were already paid for.

Re-running is NOT safe. Every row in the file becomes a row in the table, including one
that is already there — migration 0010 dropped the unique index on
`(source_url, content_sha256)` that used to make a second run a no-op, at the user's explicit
request, so a corpus that itself contained exact-content duplicates across several
ingestion runs could be loaded whole rather than have most of it silently skipped. Running
this script twice against the same file doubles every row in it. Confirm the file is right
before loading it, not after.

`scripts.polish_documents_csv` never removes a row either, so a duplicate, a short chunk, or
a row it could only flag as questionable reaches this file too, and now lands in the table
exactly as it was flagged.
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from scripts.polish_documents_csv import DEFAULT_OUTPUT, OUTPUT_COLUMNS
from src.database.session import create_cli_engine
from src.documents.constants import EMBEDDING_DIMENSIONS
from src.documents.ingestion import PendingDocument, insert_documents
from src.documents.utils import content_hash

# Rows per INSERT. A row is 1536 floats of text, so a hundred of them is a statement of a
# couple of megabytes — large enough to make the round trips cheap, small enough that the
# statement itself stays something a server will parse without complaint.
INSERT_BATCH_SIZE = 100

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_OUTPUT,
        help=f"polished CSV to load (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and validate the file without connecting to the database",
    )

    return parser.parse_args()


def read_documents(path: Path) -> list[PendingDocument]:
    """Parse the CSV into rows ready to insert, refusing the whole file on any bad one.

    All-or-nothing on purpose: a corpus half-loaded because row 200 had a short vector is a
    corpus whose gaps show up later as an answer that quietly had no source, which is the
    one failure this table exists to prevent. The file is small and the fix is cheap, so the
    right moment to complain is before anything is written.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in OUTPUT_COLUMNS if column not in (reader.fieldnames or [])]

        if missing:
            raise SystemExit(
                f"{path} is missing column(s): {', '.join(missing)}. "
                "Run python -m scripts.polish_documents_csv on the export first.",
            )

        documents: list[PendingDocument] = []
        problems: list[str] = []

        for line, row in enumerate(reader, start=2):
            embedding = json.loads(row["embedding"])

            if len(embedding) != EMBEDDING_DIMENSIONS:
                problems.append(
                    f"line {line}: {len(embedding)} dimensions, "
                    f"column is vector({EMBEDDING_DIMENSIONS})",
                )

                continue

            # The hash is half of a unique index, so it is recomputed here rather than
            # trusted: a stored hash that does not match its own text would let the same
            # chunk in twice, which is exactly what the index is there to stop.
            expected = content_hash(row["content"])

            if row["content_sha256"] != expected:
                problems.append(f"line {line}: content_sha256 does not match content")

                continue

            documents.append(
                PendingDocument(
                    title=row["title"],
                    agency=row["agency"],
                    service=row["service"],
                    content=row["content"],
                    content_sha256=expected,
                    source_url=row["source_url"],
                    source_kind=row["source_kind"],
                    embedding=embedding,
                    embedding_model=row["embedding_model"],
                ),
            )

    if problems:
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)

        raise SystemExit(f"{len(problems)} unusable row(s); nothing was loaded.")

    return documents


def _summarise(documents: list[PendingDocument]) -> None:
    print(f"\n{len(documents)} row(s) ready:")
    print(f"  by source_kind: {dict(Counter(d.source_kind for d in documents))}")
    print(f"  by model:       {dict(Counter(d.embedding_model for d in documents))}")

    for url, count in Counter(d.source_url for d in documents).most_common():
        print(f"  {count:>5}  {url}")


async def _insert_all(db: AsyncSession, documents: list[PendingDocument]) -> int:
    """
    Insert every batch in one transaction. Returns how many rows were actually written.

    With no unique index behind `insert_documents` any more, that return value is always
    `len(documents)` — every row in the file becomes a row in the table.
    """
    inserted = 0

    for start in range(0, len(documents), INSERT_BATCH_SIZE):
        batch = documents[start : start + INSERT_BATCH_SIZE]
        inserted += await insert_documents(db, batch)
        print(f"  {min(start + len(batch), len(documents))}/{len(documents)} processed")

    await db.commit()

    return inserted


async def run(*, source: Path, dry_run: bool) -> int:
    """Returns a process exit code."""
    if not source.is_file():
        print(f"{source} does not exist.", file=sys.stderr)

        return 1

    documents = read_documents(source)
    _summarise(documents)

    if dry_run:
        print("\nDry run: the database was not touched.")

        return 0

    engine = create_cli_engine()

    try:
        async with AsyncSession(engine) as db:
            inserted = await _insert_all(db, documents)
    finally:
        await engine.dispose()

    print(f"\ninserted {inserted} row(s)")

    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args()

    sys.exit(asyncio.run(run(source=Path(args.input).expanduser(), dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
