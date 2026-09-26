import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.auth.router import router as auth_router
from src.config import settings
from src.database.session import async_session_factory, engine
from src.demo import service as demo_service
from src.exceptions.handlers import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from src.logging import configure_logging
from src.me.router import router as me_router
from src.middlewares.response import response_transformer
from src.profiles.router import router as profile_router
from src.tokens.router import router as tokens_router

# Redaction is installed before anything can log: no password, token, JWT, or cookie value
# reaches a handler even if a log line asks for one.
configure_logging()

logger = logging.getLogger(__name__)


async def _set_up_demo_account() -> None:
    """
    Create or repair the demo account, when DEMO_MODE is on.

    A failure is logged loudly but does not stop the server: `/health` should still answer,
    so whoever is setting up the demo can see the instance is up and read why the account
    is not. The log line names the error type only, never its message, which can carry SQL.
    """
    try:
        async with async_session_factory() as db:
            status = await demo_service.ensure_demo_account(db)
            await db.commit()
    except Exception as exc:  # noqa: BLE001  # startup must survive a missing database
        logger.error(  # noqa: TRY400
            "Demo account setup failed (%s). Check the database is reachable and "
            "migrated (alembic upgrade head), then restart.",
            type(exc).__name__,
        )

        return

    logger.info(
        "Demo account ready (created=%s, profile_filled=%s, token_installed=%s)",
        status.user_created,
        status.profile_filled,
        status.token_installed,
    )

    if not status.token_configured:
        logger.warning(
            "DEMO_TOKEN is not set: the demo account can log in, but has no pre-issued "
            "extension token. Issue one through the web app, or set DEMO_TOKEN in .env.",
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if settings.demo_mode:
        await _set_up_demo_account()

    yield

    await engine.dispose()


# Initialise application server
app = FastAPI(lifespan=lifespan)


app.add_exception_handler(
    HTTPException,
    http_exception_handler,  # type: ignore  # noqa: PGH003
)

# FastAPI's default for a rejected body is 422 with its own shape. The published contract
# says INVALID_REQUEST with 400, so it is overridden here.
app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,  # type: ignore  # noqa: PGH003
)

app.add_exception_handler(
    Exception,
    general_exception_handler,
)

# Transform all responses to a standard format
app.middleware("http")(response_transformer)

# Added last, so it wraps everything else and an error response still carries the CORS
# headers the browser needs to let the web app read it.
#
# Exact origins with credentials. Never `*`: browsers reject that combination outright, and
# a wildcard alongside credentials is a credential-theft amplifier.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(tokens_router)
app.include_router(profile_router)
app.include_router(me_router)


@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    """
    Liveness, plus whether this instance is running the demo account.

    `demo_mode` is reported rather than kept quiet: an instance with a known password and a
    long-lived token should say so out loud.
    """
    return {"status": "Okay", "demo_mode": settings.demo_mode}
