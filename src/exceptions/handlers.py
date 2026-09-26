import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.constants import SAFE_MESSAGE_MARKER, ErrorCode

logger = logging.getLogger(__name__)


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    logger.error(
        "HTTP exception: %s %s - %s",
        request.method,
        request.url.path,
        exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    A rejected body becomes INVALID_REQUEST / 400, not FastAPI's default 422.

    Only the *names* of the offending fields are reported. Pydantic's error objects carry an
    `input` key holding the value that failed, which for a signup body is the password — so
    `exc.errors()` must never be serialised into a response or a log line.
    """
    safe = _safe_message(exc)

    if safe is not None:
        logger.warning(
            "Request rejected: %s %s - %s",
            request.method,
            request.url.path,
            safe,
        )

        return _invalid_request(safe)

    fields = sorted(
        {
            str(part)
            for error in exc.errors()
            for part in error.get("loc", ())
            if part not in {"body", "query", "path", "header"}
        },
    )

    logger.warning(
        "Request rejected: %s %s - fields: %s",
        request.method,
        request.url.path,
        fields,
    )

    detail = "Some details were missing or invalid."

    if fields:
        detail = f"Please check these and try again: {', '.join(fields)}."

    return _invalid_request(detail)


def _safe_message(exc: RequestValidationError) -> str | None:
    """
    The first validator message explicitly marked as safe to show, if there is one.

    A validator that wants to explain itself — rather than have its field named in a list —
    raises `ValueError(f"{SAFE_MESSAGE_MARKER} …")`. Only the literal behind the marker is
    returned, and nothing from the error's `input`, so a rejected password cannot ride out
    on a message.
    """
    for error in exc.errors():
        message = str(error.get("msg", ""))

        if SAFE_MESSAGE_MARKER in message:
            return message.split(SAFE_MESSAGE_MARKER, maxsplit=1)[1].strip()

    return None


def _invalid_request(message: str) -> JSONResponse:
    """A 400 in the error contract's shape."""
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": ErrorCode.INVALID_REQUEST,
                "message": message,
            },
        },
    )


async def general_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    """
    500, in the contract shape, with nothing from the exception in the body.

    An unreachable database lands here. The message says what to do, never what failed:
    the exception's text can carry SQL or a connection string.
    """
    logger.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": ErrorCode.INTERNAL,
                "message": "Something went wrong on our side. Please try again shortly.",
            },
        },
    )
