"""
Logging with redaction.

Log metadata, never content: field ids, counts, durations, statuses, outcomes. Never a
profile value, never page text, never a prompt containing user data, never a credential.

This filter is the backstop, not the rule. Code is still written not to log secrets in the
first place; this catches the line someone adds in a hurry at 2am during a demo build.
"""

import logging
import re
from typing import Any

REDACTED = "[REDACTED]"

# Keys whose values are removed wherever they appear in a structured log argument.
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "raw_password",
        "token",
        "raw_token",
        "token_hash",
        "authorization",
        "cookie",
        "set-cookie",
        "jwt",
        "secret",
        "jwt_secret",
        "password_hash",
    },
)

# Anything shaped like one of our tokens, a JWT, or a cookie header value, even inside a
# message that was assembled as a plain string.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ngv_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}"),
    re.compile(r"\$argon2[a-z]{1,4}\$[^\s\"']+"),
    re.compile(r"(?i)(authorization|cookie)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(password|secret|token)\s*[:=]\s*\S+"),
)


def _scrub_text(value: str) -> str:
    for pattern in _PATTERNS:
        value = pattern.sub(REDACTED, value)

    return value


def _scrub(value: Any) -> Any:  # noqa: ANN401  # log args are arbitrary
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {
            key: (REDACTED if str(key).lower() in SENSITIVE_KEYS else _scrub(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return type(value)(_scrub(item) for item in value)

    return value


class RedactingFilter(logging.Filter):
    """
    Scrubs every record that passes through.

    The record is rendered first and the result replaces `msg`, with `args` cleared. Scrubbing
    the format string on its own is not safe: a pattern can swallow a `%s` placeholder and
    leave its argument behind, and the record then raises when anything tries to format it —
    losing the log line entirely.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = _scrub(record.args)

        try:
            rendered = record.getMessage()
        except (TypeError, ValueError):
            # A malformed log call. Keep the record but let nothing through unscrubbed.
            record.msg = _scrub_text(str(record.msg))
            record.args = ()

            return True

        record.msg = _scrub_text(rendered)
        record.args = ()

        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Install the redaction filter on the root logger's handlers."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")

    redactor = RedactingFilter()
    root = logging.getLogger()

    for handler in root.handlers:
        handler.addFilter(redactor)
