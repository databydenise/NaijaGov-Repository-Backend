"""Workflow helpers with no database access."""

import re

# Hosts that only mean anything on a developer's machine.
LOCAL_HOST_NAMES = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "[::1]"})  # noqa: S104

# Longer than any URL pattern a portal needs, short enough that nothing here can spend real
# time. A pattern that wants more than this is doing something a second workflow should do.
MAX_PATTERN_LENGTH = 200

_QUANTIFIERS = frozenset("+*{")

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


def _risky_group_body(pattern: str) -> str | None:
    """
    The body of the first quantified group that can backtrack catastrophically, or None.

    Walks the pattern tracking escapes and character classes, so a `(` inside `[(]` or
    after a backslash is not read as a group. For every group followed by `+`, `*` or `{`,
    the body is unsafe if it holds either its own quantifier (`(a+)+`) or an alternation
    (`(a|aa)+`). Both make the engine try exponentially many ways to split the same input
    when the overall match fails, and `re` cannot be interrupted once it starts.

    Conservative: a quantified group with an alternation is rejected even when its branches
    do not overlap. A URL pattern that needs one can be written as two patterns.
    """
    stack: list[int] = []
    escaped = False
    in_class = False

    for index, char in enumerate(pattern):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "(":
            stack.append(index)
        elif char == ")" and stack:
            start = stack.pop()
            following = pattern[index + 1 : index + 2]

            if following not in _QUANTIFIERS:
                continue

            body = pattern[start + 1 : index]

            if _has_bare(body, _QUANTIFIERS) or _has_bare(body, {"|"}):
                return body

    return None


def _has_bare(body: str, characters: set[str] | frozenset[str]) -> bool:
    """Whether `body` holds one of these characters outside an escape or a class."""
    escaped = False
    in_class = False

    for char in body:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char in characters:
            return True

    return False


def pattern_rejection_reason(pattern: str) -> str | None:
    """
    Why this `url_patterns` entry is unsafe to run, or None if it is fine.

    Two checks, both cheap, both about what a pattern can cost rather than what it means:
    it must be short, and it must not nest a quantifier inside a quantified group. The
    second is what makes `(a+)+$` take exponential time on a string that does not match,
    and `re` gives us no way to interrupt it once it starts.

    The seed refuses a pattern this rejects; the read path skips one, because a single bad
    row should not take `/context` down for every other workflow.
    """
    if not pattern:
        return "is empty"

    if len(pattern) > MAX_PATTERN_LENGTH:
        return f"is longer than {MAX_PATTERN_LENGTH} characters"

    if _risky_group_body(pattern) is not None:
        return (
            "repeats a group that holds a quantifier or an alternation, which can "
            "backtrack forever"
        )

    try:
        re.compile(pattern)
    except re.error as exc:
        return f"is not a valid regex ({exc})"

    return None


def is_local_host(host: str) -> bool:
    """Whether a host only resolves on the machine running the code, port ignored."""
    name = host.rsplit(":", maxsplit=1)[0] if not host.endswith("]") else host

    return name in LOCAL_HOST_NAMES
