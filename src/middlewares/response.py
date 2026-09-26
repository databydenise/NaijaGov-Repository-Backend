import json
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import Request, Response
from starlette.responses import JSONResponse

RESPONSE_LOWER_LIMIT = 200
RESPONSE_UPPER_LIMIT = 299

# Recomputed by the new response for the rewritten body. Everything else the route set —
# Set-Cookie above all, but also any cache or custom header — is carried across.
RECOMPUTED_HEADERS = frozenset({b"content-length", b"content-type"})


async def response_transformer(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response: Response = await call_next(request)

    success = RESPONSE_LOWER_LIMIT <= response.status_code <= RESPONSE_UPPER_LIMIT

    # Only transform JSON responses
    content_type = response.headers.get("content-type", "")

    if "application/json" not in content_type:
        return response

    body: Literal[b""] = b""

    async for chunk in response.body_iterator:  # pyright: ignore[reportAttributeAccessIssue]
        body += chunk

    data = json.loads(body)

    content = {"success": True, "data": data} if success else {"success": False, **data}

    wrapped = JSONResponse(
        status_code=response.status_code,
        content=content,
    )

    # A fresh response starts with no headers of its own beyond these two, so without this
    # the login cookie is silently dropped. Raw headers, because one response can carry
    # several Set-Cookie lines and a dict would keep only the last.
    wrapped.raw_headers.extend(
        (name, value)
        for name, value in response.raw_headers
        if name.lower() not in RECOMPUTED_HEADERS
    )

    return wrapped
