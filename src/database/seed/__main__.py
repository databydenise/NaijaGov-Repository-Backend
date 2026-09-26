"""CLI: `python -m src.database.seed [--reset-reference]`."""

import argparse
import asyncio
import logging
import sys

from src.database.seed.exceptions import SeedValidationError
from src.database.seed.run import seed

logger = logging.getLogger("src.database.seed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.database.seed",
        description="Load workflows, steps, and rules. Safe to run repeatedly.",
    )
    parser.add_argument(
        "--reset-reference",
        action="store_true",
        help=(
            "Delete and reload workflows, steps, and rules before writing. "
            "Use after editing a seed file so removed rows actually go. "
            "Never touches users, profiles, tokens, sessions, or the action log."
        ),
    )

    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    try:
        written = asyncio.run(seed(reset_reference=args.reset_reference))
    except SeedValidationError as exc:
        logger.error("Seed failed: %s", exc)  # noqa: TRY400  # a traceback adds nothing here

        return 1

    for table, count in written.items():
        logger.info("%s: %d rows", table, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
