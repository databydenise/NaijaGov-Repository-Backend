"""
Turn an exported `documents` CSV into one the loader can read.

    python -m scripts.polish_documents_csv ~/Downloads/documents_rows.csv
    python -m scripts.polish_documents_csv raw.csv --output data/documents_corpus.csv

The export carries `id,title,agency,service,context,source_url,embedding` — the text and the
vectors, but none of the four columns the table needs to keep a corpus honest. This fills
them in for every row in the file, in the same order, with the same count. Nothing is
dropped, however it looks:

  * `content`         <- `context`, verbatim. The embedding was computed from this exact
                         text, so rewriting it would leave the vector describing something
                         the row no longer says.
  * `content_sha256`  <- `src.documents.utils.content_hash`, the same function ingestion
                         uses, so a later `python -m scripts.ingest` recognises these rows
                         as already present instead of embedding them again.
  * `source_kind`     <- derived from the URL, cross-checked against the title.
  * `embedding_model` <- `--embedding-model`, defaulting to the configured one.

A row that looks wrong — a duplicate, a chunk too short to be useful, an unusable embedding,
a blank required column — is written anyway and named in the "flagged for review" report.
Deciding what to do with it is a person's job, made by reading that report; this script's
job is only to point at it, not to make the call by leaving it out. The database's own
unique index is where a genuine duplicate is actually resolved, at load time.

No network, no database, no OpenAI key. Pure CSV in, CSV out, exact row count preserved.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.documents.constants import EMBEDDING_DIMENSIONS, SourceKind
from src.documents.utils import MIN_CHUNK_CHARS, content_hash, normalize_whitespace

DEFAULT_OUTPUT = "data/documents_corpus.csv"

# The export's own column names. `context` is this table's `content` under another name.
RAW_CONTENT_COLUMN = "context"
RAW_COLUMNS = ("id", "title", "agency", "service", RAW_CONTENT_COLUMN, "source_url", "embedding")

# What the loader reads: every insertable column on `documents`, in table order. `id` is the
# table's own autoincrement and `ingested_at` its server default, so neither belongs in a
# file that describes content — a row's ingest date is when it entered this database, not a
# number an export happened to carry.
OUTPUT_COLUMNS = (
    "title",
    "agency",
    "service",
    "content",
    "content_sha256",
    "source_url",
    "source_kind",
    "embedding",
    "embedding_model",
)

# Columns the table declares NOT NULL. A blank one is flagged, not fixed — this script does
# not invent a title or a URL that was not in the export.
REQUIRED_TEXT_COLUMNS = ("title", "agency", "service", "source_url")


@dataclass(frozen=True)
class Flag:
    """One row worth a second look, kept for the report. Never removed from the output."""

    row_id: str
    reason: str
    detail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="the exported CSV to read")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"where to write the polished CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help=(
            "value for the embedding_model column. Defaults to the configured "
            f"{settings.embedding_model}. Set it to what actually produced these vectors: "
            "a wrong name either hides a mixed corpus from scripts.reindex or makes it "
            "re-embed every row at a cost."
        ),
    )

    return parser.parse_args()


def derive_source_kind(source_url: str, title: str) -> tuple[str, str | None]:
    """
    Work out `faq | page | pdf` from the URL. Returns the kind and a warning, if any.

    The URL decides, because it is the column the row is constrained on and the one a
    citation shows. The title only gets a say as a cross-check: ingestion writes `FAQ: …`
    and `… (page N)` prefixes, so a title that disagrees with the URL means one of the two
    is wrong and a person should look before loading.
    """
    path = source_url.split("?", 1)[0].split("#", 1)[0].lower()

    if path.endswith(".pdf"):
        kind = SourceKind.PDF
    elif "faq" in path:
        kind = SourceKind.FAQ
    else:
        kind = SourceKind.PAGE

    lowered = title.lower()
    claimed = None

    if lowered.startswith("faq:"):
        claimed = SourceKind.FAQ
    elif "(page " in lowered:
        claimed = SourceKind.PDF

    if claimed and claimed != kind:
        return kind, f"title looks like {claimed}, URL says {kind}"

    return kind, None


def _embedding_error(raw: str) -> str | None:
    """
    Why this embedding looks unusable, or None if it looks fine. Reported, not enforced.

    This never removes the row it is describing — it only decides what goes in the report,
    so a person can see a bad vector coming instead of discovering it in a later load.
    """
    try:
        vector = json.loads(raw)
    except (TypeError, ValueError):
        return "not valid JSON"

    if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
        return "not a list of numbers"

    if len(vector) != EMBEDDING_DIMENSIONS:
        return f"{len(vector)} dimensions, column is vector({EMBEDDING_DIMENSIONS})"

    return None


def read_raw_rows(path: Path) -> list[dict[str, str]]:
    """Read the export, failing loudly if it is not the shape this script was written for."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in RAW_COLUMNS if column not in (reader.fieldnames or [])]

        if missing:
            message = f"{path} is missing column(s): {', '.join(missing)}"

            raise SystemExit(message)

        return list(reader)


def _duplicate_flags(rows: list[dict[str, str]]) -> list[Flag]:
    """
    Flag rows that share a `(source_url, content_sha256)` key, without removing either one.

    That pair is the table's own unique index. Loading two rows that share it is not an
    error — `scripts.load_documents` inserts the first and the database's `ON CONFLICT DO
    NOTHING` silently skips the rest — so this is a heads-up, not a gate.
    """
    ids_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)

    for row in rows:
        key = (row["source_url"], content_hash(row[RAW_CONTENT_COLUMN]))
        ids_by_key[key].append(row["id"])

    return [
        Flag(
            row_id=row_id,
            reason="duplicate content",
            detail=f"same text as id(s) {', '.join(i for i in ids if i != row_id)}",
        )
        for ids in ids_by_key.values()
        if len(ids) > 1
        for row_id in ids
    ]


def polish(
    rows: list[dict[str, str]],
    *,
    embedding_model: str,
) -> tuple[list[dict[str, str]], list[Flag]]:
    """
    Fill the missing columns for every row. Returns them all, in order, plus what to review.

    The output list is always the same length as `rows`, in the same order: nothing here
    decides a row does not belong in the corpus. That call, if it needs making, belongs to
    whoever reads the flags this returns alongside it.
    """
    flags = _duplicate_flags(rows)
    polished: list[dict[str, str]] = []

    for row in rows:
        row_id = row["id"]
        content = row[RAW_CONTENT_COLUMN]
        normalized = normalize_whitespace(content)

        blank = [column for column in REQUIRED_TEXT_COLUMNS if not row[column].strip()]

        if blank:
            flags.append(Flag(row_id, "blank required column", ", ".join(blank)))

        # Ingestion never stores a chunk this short — these read like a chunker's leftovers,
        # "s.", "nd records:" — but that is a reason to look at the row, not to leave it out.
        if len(normalized) < MIN_CHUNK_CHARS:
            detail = f"{len(normalized)} < {MIN_CHUNK_CHARS} chars"
            flags.append(Flag(row_id, "content too short", detail))

        embedding_error = _embedding_error(row["embedding"])

        if embedding_error:
            flags.append(Flag(row_id, "unusable embedding", embedding_error))

        source_kind, mismatch = derive_source_kind(row["source_url"], row["title"])

        if mismatch:
            flags.append(Flag(row_id, "title/URL mismatch", mismatch))

        polished.append(
            {
                "title": row["title"],
                "agency": row["agency"],
                "service": row["service"],
                "content": content,
                "content_sha256": content_hash(content),
                "source_url": row["source_url"],
                "source_kind": source_kind,
                # Written back exactly as it was read. Re-serialising the parsed floats
                # would round every value for no reason and make the file undiffable.
                "embedding": row["embedding"],
                "embedding_model": embedding_model,
            },
        )

    return polished, flags


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict[str, str]], flags: list[Flag], output: Path, *, read: int) -> None:
    if flags:
        print(f"\n{len(flags)} row(s) flagged for review — none of them were removed:")

        for reason, count in Counter(flag.reason for flag in flags).most_common():
            print(f"  {count:>4}  {reason}")
            for flag in (flag for flag in flags if flag.reason == reason):
                print(f"          id {flag.row_id}: {flag.detail}")

    print(f"\nby source_kind: {dict(Counter(row['source_kind'] for row in rows))}")
    print(f"by service:     {dict(Counter(row['service'] for row in rows))}")
    print(f"\nwrote {len(rows)} row(s) to {output} (read {read}; none dropped)")


def run(*, source: Path, output: Path, embedding_model: str) -> int:
    """Returns a process exit code."""
    if not source.is_file():
        print(f"{source} does not exist.", file=sys.stderr)

        return 1

    raw = read_raw_rows(source)
    print(f"read {len(raw)} row(s) from {source}")

    rows, flags = polish(raw, embedding_model=embedding_model)

    write_rows(rows, output)
    _report(rows, flags, output, read=len(raw))

    return 0


def main() -> None:
    args = _parse_args()

    sys.exit(
        run(
            source=Path(args.input).expanduser(),
            output=Path(args.output).expanduser(),
            embedding_model=args.embedding_model,
        ),
    )


if __name__ == "__main__":
    main()
