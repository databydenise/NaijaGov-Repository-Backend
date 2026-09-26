"""
A cap on request body size, per path.

Pydantic's field caps only apply once a body has been read and parsed, which is too late:
a 50 MB snapshot costs the memory and the parse before any of them fire. This refuses it
at the transport, before the route is reached.

Written against the ASGI interface rather than as a `BaseHTTPMiddleware`, because it has
to inspect and short-circuit the request stream, and it must emit a response the
JSON-wrapping middleware does not touch.
"""

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from src.constants import ErrorCode
from src.context.constants import MAX_BODY_BYTES

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

# Path prefix to its limit. Only the routes that take a page snapshot need one; the rest
# take a handful of fields and are capped by their own models.
BODY_LIMITS: dict[str, int] = {"/context": MAX_BODY_BYTES}

# 413. The body is already in the wrapped shape the response middleware would produce,
# because this response is sent directly and never passes through it.
_TOO_LARGE_BODY = json.dumps(
    {
        "success": False,
        "error": {
            "code": ErrorCode.INVALID_REQUEST,
            "message": "That page is too large for me to read. Try a simpler page.",
        },
    },
).encode("utf-8")


def _limit_for(path: str) -> int | None:
    """The limit covering this path, if any."""
    for prefix, limit in BODY_LIMITS.items():
        if path.startswith(prefix):
            return limit

    return None


async def _send_too_large(send: Send) -> None:
    """Reply 413 and stop. The request body is deliberately not drained."""
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_TOO_LARGE_BODY)).encode("ascii")),
            ],
        },
    )
    await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})


class BodyLimitMiddleware:
    """Refuses a request body over the limit for its path."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)

            return

        limit = _limit_for(scope.get("path", ""))

        if limit is None:
            await self.app(scope, receive, send)

            return

        declared = _declared_length(scope)

        if declared is not None and declared > limit:
            logger.warning(
                "Body over limit on %s: declared %d bytes, limit %d",
                scope.get("path", ""),
                declared,
                limit,
            )
            await _send_too_large(send)

            return

        # A chunked body declares no length, so the only way to know is to count. The
        # counter replaces `receive`, so the route reads the same stream and nothing is
        # buffered here.
        await self.app(scope, _counting_receive(receive, limit, send), send)


def _declared_length(scope: Scope) -> int | None:
    """The request's `Content-Length`, or None when it is absent or unreadable."""
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue

        try:
            return int(value)
        except ValueError:
            return None

    return None


def _counting_receive(receive: Receive, limit: int, send: Send) -> Receive:
    """
    Wrap `receive` so a body that grows past the limit is cut off.

    On exceeding it we send the 413 ourselves and hand the application an empty final
    chunk. The route's parse then fails on a truncated body, but the client already has
    the right answer — and the alternative is reading an unbounded stream to be polite
    about it.
    """
    total = 0
    responded = False

    async def counting() -> MutableMapping[str, Any]:
        nonlocal total, responded

        message = await receive()

        if message["type"] != "http.request":
            return message

        total += len(message.get("body", b""))

        if total > limit and not responded:
            responded = True
            logger.warning("Streamed body over limit: %d bytes, limit %d", total, limit)
            await _send_too_large(send)

            return {"type": "http.request", "body": b"", "more_body": False}

        return message

    return counting
