import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

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

    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": "INVALID_REQUEST",
                "message": detail,
            },
        },
    )


async def general_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
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
                "message": "Internal server error",
            },
        },
    )
