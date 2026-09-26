"""
Shared error codes.

One shape everywhere: `{"error": {"code": ..., "message": ...}}`, wrapped by the response
middleware. `message` is written for the user, because the extension's side panel shows it
verbatim — plain English, no jargon, and a next step where one exists.
"""

from typing import Final

# `/health`'s count of demo rules. Here rather than in `workflows/` because the reporter is
# `main.py` and the counted table belongs to `knowledge/`.
PLACEHOLDER_RULES_CACHE_KEY = "knowledge:placeholder_rules"

# Marks a validation message as one we wrote ourselves, so the exception handler may show
# it to the caller verbatim. The handler otherwise reports field names only, because
# pydantic's error objects carry the rejected input and that input can be a password.
# A message behind this marker must therefore be a fixed literal that never interpolates
# a submitted value.
SAFE_MESSAGE_MARKER = "[safe]"



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
