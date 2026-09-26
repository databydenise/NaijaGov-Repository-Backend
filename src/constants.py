"""
Shared error codes.

One shape everywhere: `{"error": {"code": ..., "message": ...}}`, wrapped by the response
middleware. `message` is written for the user, because the extension's side panel shows it
verbatim — plain English, no jargon, and a next step where one exists.
"""

from typing import Final


class ErrorCode:
    """Codes callers may branch on. Adding one is a change to the published contract."""

    UNAUTHENTICATED: Final = "UNAUTHENTICATED"
    INVALID_CREDENTIALS: Final = "INVALID_CREDENTIALS"
    EMAIL_TAKEN: Final = "EMAIL_TAKEN"
    INVALID_REQUEST: Final = "INVALID_REQUEST"
    NOT_FOUND: Final = "NOT_FOUND"
    RATE_LIMITED: Final = "RATE_LIMITED"
    PAGE_CHANGED: Final = "PAGE_CHANGED"
    MODEL_UNAVAILABLE: Final = "MODEL_UNAVAILABLE"
    INTERNAL: Final = "INTERNAL"
