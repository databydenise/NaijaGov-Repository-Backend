"""Workflow helpers with no database access."""

import re

# Hosts that only mean anything on a developer's machine.
LOCAL_HOST_NAMES = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "[::1]"})  # noqa: S104

# `^https://`, `https://`, `^https?://`, and the same for http, with or without the anchor.
_SCHEME = re.compile(r"^\^?https?(?:\?)?://")

# What a host is allowed to look like once unescaped: letters, digits, dots and hyphens,
# with an optional port. Anything else means the pattern uses regex in the host itself.
_PLAIN_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::\d{1,5})?$")

# Escapes that stand for a literal character in a host.
_LITERAL_ESCAPES = {r"\.": ".", r"\-": "-", r"\:": ":"}


def host_from_pattern(pattern: str) -> str | None:
    """
    The literal host a `url_patterns` regex matches, or None if it has none.

    `^https://services\\.example\\.gov\\.ng/business-name/register` gives
    `services.example.gov.ng`. A pattern with regex in its host (`.*\\.gov\\.ng`, a
    character class, an alternation) gives None and is skipped: telling the panel a host
    is supported when it is only a guess is worse than leaving it off.
    """
    unescaped = pattern.replace(r"\/", "/")
    scheme = _SCHEME.match(unescaped)

    if scheme is None:
        return None

    rest = unescaped[scheme.end() :]
    host = re.split(r"[/$]", rest, maxsplit=1)[0]

    for escaped, literal in _LITERAL_ESCAPES.items():
        host = host.replace(escaped, literal)

    host = host.lower()

    if not _PLAIN_HOST.match(host):
        return None

    return host


def is_local_host(host: str) -> bool:
    """Whether a host only resolves on the machine running the code, port ignored."""
    name = host.rsplit(":", maxsplit=1)[0] if not host.endswith("]") else host

    return name in LOCAL_HOST_NAMES
