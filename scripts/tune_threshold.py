"""
Choose the retrieval distance cutoff by measurement.

    python -m scripts.tune_threshold

Embeds each labelled query in `data/retrieval_queries.yaml` once, records the distance to
its nearest chunk, and then sweeps candidate cutoffs over those recorded distances. One
embedding per query, not one per cutoff: the distances do not change as the cutoff moves,
only the verdict does.

What to pick: **the loosest cutoff at which no unanswerable query returns a chunk.** Recall
on the answerable set is the thing being traded away, and trading it away is correct here.
A missing answer is a shrug. A confident wrong fee on a government form is the failure this
whole package exists to prevent, and it is what the prototype actually produced.

The script prints the recommendation but does not apply it. Setting the number is a commit:
`DEFAULT_RETRIEVAL_MAX_DISTANCE` in `src/documents/constants.py`, with the date and the
corpus size it was measured against.
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.session import create_cli_engine
from src.database.session import engine as app_engine
from src.documents import embeddings
from src.documents.constants import MAX_SEARCH_LIMIT, TUNING_CUTOFFS
from src.documents.health import get_corpus_stats
from src.documents.service import search_government_information

DEFAULT_FIXTURES = "data/retrieval_queries.yaml"

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"

# Every candidate distance has to be measured against something, so the sweep searches with
# the cutoff wide open and applies the candidates afterwards. 2.0 is the maximum a cosine
# distance can be, so nothing is filtered out by the query itself.
NO_CUTOFF = 2.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Measurement:
    """One labelled query and how close the corpus came to answering it."""

    query: str
    label: str

    # None when the search returned nothing even with the cutoff wide open, which means the
    # corpus is empty or the search failed.
    best_distance: float | None

    @property
    def is_answerable(self) -> bool:
        return self.label == ANSWERABLE


@dataclass(frozen=True)
class Verdict:
    """How one candidate cutoff scores over the whole fixture set."""

    cutoff: float

    # Answerable queries that still return something. Recall.
    true_positives: int

    # Unanswerable queries that return something anyway. The number that must reach zero.
    false_positives: int

    # Answerable queries that come back empty. The acceptable failure.
    false_negatives: int

    @property
    def precision(self) -> float:
        returned = self.true_positives + self.false_positives

        return self.true_positives / returned if returned else 1.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives

        return self.true_positives / total if total else 0.0

    @property
    def is_safe(self) -> bool:
        """No unanswerable query returns a chunk. The only hard requirement."""
        return self.false_positives == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", default=DEFAULT_FIXTURES, help="labelled query file"
    )

    return parser.parse_args()


def load_fixtures(path: str) -> list[tuple[str, str]]:
    """
    Read the labelled queries, refusing anything unlabelled.

    A query with no label would quietly count as neither, which makes a sweep look cleaner
    than it is.
    """
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(rows, list) or not rows:
        message = f"{path} must hold a non-empty list of queries"
        raise ValueError(message)

    fixtures = []

    for index, row in enumerate(rows, start=1):
        query = (row or {}).get("query")
        label = (row or {}).get("label")

        if not query or label not in (ANSWERABLE, UNANSWERABLE):
            message = (
                f"{path} entry {index} needs a query and a label of "
                f"'{ANSWERABLE}' or '{UNANSWERABLE}'"
            )
            raise ValueError(message)

        fixtures.append((query, label))

    return fixtures


async def measure(fixtures: list[tuple[str, str]]) -> list[Measurement]:
    """The nearest-chunk distance for every fixture, searched with the cutoff wide open."""
    measurements = []

    for query, label in fixtures:
        result = await search_government_information(
            query,
            limit=MAX_SEARCH_LIMIT,
            max_distance=NO_CUTOFF,
        )

        if not result.available:
            message = (
                "retrieval reported unavailable; check OPENAI_API_KEY and the database"
            )
            raise RuntimeError(message)

        best = result.chunks[0].distance if result.chunks else None
        measurements.append(Measurement(query=query, label=label, best_distance=best))

    return measurements


def score(measurements: list[Measurement], cutoff: float) -> Verdict:
    """How one cutoff would have judged every measured query."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for measurement in measurements:
        returned = (
            measurement.best_distance is not None and measurement.best_distance < cutoff
        )

        if measurement.is_answerable and returned:
            true_positives += 1
        elif measurement.is_answerable:
            false_negatives += 1
        elif returned:
            false_positives += 1

    return Verdict(
        cutoff=cutoff,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def recommend(verdicts: list[Verdict]) -> Verdict | None:
    """
    The loosest cutoff that still lets no unanswerable query through.

    Loosest rather than strictest among the safe ones: every safe cutoff is equally safe by
    the hard requirement, so the one to take is the one that keeps the most real answers.
    """
    safe = [verdict for verdict in verdicts if verdict.is_safe]

    if not safe:
        return None

    return max(safe, key=lambda verdict: verdict.cutoff)


def _print_distances(measurements: list[Measurement]) -> None:
    """
    Every query's nearest distance, worst first within each label.

    Printed because the table is what a person reads to see whether the two labels actually
    separate. If the closest unanswerable query sits nearer than the furthest answerable
    one, no cutoff can be both safe and useful, and the answer is more corpus rather than a
    different number.
    """
    for label in (ANSWERABLE, UNANSWERABLE):
        print(f"\n  {label}:")
        rows = [m for m in measurements if m.label == label]

        for measurement in sorted(rows, key=lambda m: m.best_distance or 9.0):
            distance = (
                f"{measurement.best_distance:.4f}"
                if measurement.best_distance is not None
                else "  none"
            )
            print(f"    {distance}  {measurement.query}")


def _print_sweep(verdicts: list[Verdict]) -> None:
    print("\n  cutoff  returns  missed  false+  precision  recall  safe")

    for verdict in verdicts:
        print(
            f"    {verdict.cutoff:.2f}"
            f"  {verdict.true_positives:>7}"
            f"  {verdict.false_negatives:>6}"
            f"  {verdict.false_positives:>6}"
            f"  {verdict.precision:>9.2f}"
            f"  {verdict.recall:>6.2f}"
            f"  {'yes' if verdict.is_safe else 'NO':>4}",
        )


async def run(fixtures_path: str) -> int:
    """Returns a process exit code."""
    if not embeddings.is_configured():
        print(
            "OPENAI_API_KEY is not set, so no query can be embedded.", file=sys.stderr
        )

        return 1

    try:
        fixtures = load_fixtures(fixtures_path)
    except (OSError, ValueError) as exc:
        print(f"Could not read {fixtures_path}: {exc}", file=sys.stderr)

        return 1

    # Two engines are in play: a throwaway one for the corpus count, and the application's
    # own pool, which `search_government_information` opens its own sessions from. Both are
    # disposed here so the script exits without leaving a connection behind.
    engine = create_cli_engine()

    try:
        async with AsyncSession(engine) as db:
            stats = await get_corpus_stats(db)

        if not stats.documents:
            print(
                "The documents table is empty. Run `python -m scripts.ingest` first.",
                file=sys.stderr,
            )

            return 1

        print(
            f"Corpus: {stats.documents} chunks from {stats.distinct_sources} source(s), "
            f"model {settings.embedding_model}",
        )
        print(f"Fixtures: {len(fixtures)} queries from {fixtures_path}")

        measurements = await measure(fixtures)
    finally:
        await engine.dispose()
        await app_engine.dispose()

    _print_distances(measurements)

    verdicts = [score(measurements, cutoff) for cutoff in TUNING_CUTOFFS]
    _print_sweep(verdicts)

    best = recommend(verdicts)

    if best is None:
        print(
            "\nNo candidate cutoff keeps every unanswerable query out. The corpus overlaps "
            "the unanswerable set more than a distance threshold can separate — widen the "
            "corpus or re-check the labels rather than picking a number from this table.",
        )

        return 1

    print(
        f"\nRecommended RETRIEVAL_MAX_DISTANCE: {best.cutoff:.2f}"
        f"  (recall {best.recall:.0%}, no false positives)",
    )
    print(
        "Record it in src/documents/constants.py as DEFAULT_RETRIEVAL_MAX_DISTANCE, "
        "with today's date and the corpus size above.",
    )

    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s %(message)s"
    )
    args = _parse_args()

    sys.exit(asyncio.run(run(args.fixtures)))


if __name__ == "__main__":
    main()
