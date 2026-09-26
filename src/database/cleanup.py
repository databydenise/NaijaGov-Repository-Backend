"""CLI: `python -m src.database.cleanup` — delete expired sessions.

Queries filter on `expires_at > now()` regardless of whether this has run, so a missed
sweep is a storage cost and never a correctness bug. Logs a count and nothing else: no user
id, no page hash, no history.

No `pg_cron` schedule is installed. If one is wanted it goes in its own migration.
"""

import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.database.session import create_cli_engine
from src.sessions.service import delete_expired_sessions

logger = logging.getLogger("src.database.cleanup")


async def run_cleanup() -> int:
    """Delete expired sessions. Returns how many rows went."""
    engine = create_cli_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            deleted = await delete_expired_sessions(db)
            await db.commit()
    finally:
        await engine.dispose()

    return deleted


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    deleted = asyncio.run(run_cleanup())
    logger.info("Deleted %d expired session(s)", deleted)

    return 0


if __name__ == "__main__":
    sys.exit(main())
