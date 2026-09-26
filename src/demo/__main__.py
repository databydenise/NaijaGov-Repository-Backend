"""
CLI: `python -m src.demo` — optional. Prints the demo credentials for the judging sheet.

Not needed to run the demo: with `DEMO_MODE=true`, `fastapi dev src/main.py` sets the account
up at startup. This runs the same idempotent setup, then prints what a judge needs, so the
sheet can be filled in without opening `.env`.
"""

import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.config import settings
from src.database.session import create_cli_engine
from src.demo.exceptions import DemoModeDisabled, DemoTokenConflict
from src.demo.service import DemoAccountStatus, ensure_demo_account
from src.logging import configure_logging

logger = logging.getLogger("src.demo")


async def _set_up() -> DemoAccountStatus:
    engine = create_cli_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            status = await ensure_demo_account(db)
            await db.commit()
    finally:
        await engine.dispose()

    return status


def main() -> int:
    configure_logging()

    try:
        asyncio.run(_set_up())
    except (DemoModeDisabled, DemoTokenConflict) as exc:
        logger.error("%s", exc)  # noqa: TRY400  # a traceback adds nothing here

        return 1

    # Printed, never logged: the redaction filter would scrub the token from a log line,
    # which is exactly what we want everywhere except here.
    print("\nDemo account")  # noqa: T201
    print(f"  email:    {settings.demo_email}")  # noqa: T201
    print(f"  password: {settings.demo_password}")  # noqa: T201
    print(f"  token:    {settings.demo_token or '(not set — add DEMO_TOKEN to .env)'}")  # noqa: T201
    print()  # noqa: T201

    return 0


if __name__ == "__main__":
    sys.exit(main())
