"""
Pure helpers for `/context`: the page hash, and the host for a log line.

No database, no HTTP, no clock.
"""

import hashlib
from collections.abc import Sequence
from urllib.parse import urlsplit

from src.context.schemas import PageField
from src.knowledge.normalize import normalize_label

# Separators chosen so no label can forge a boundary: a label with a tab in it is
# normalised away before it gets here.
_FIELD_SEPARATOR = "\t"
_TRIPLE_SEPARATOR = "\n"


def compute_page_hash(url: str, fields: Sequence[PageField]) -> str:
    """
    A stable fingerprint of what this page asks for.

    Built from the URL path and the `(field_id, normalised label, type)` triples in page
    order. Deliberately excludes the query string, the title and the headings: a portal
    that appends a tracking parameter or re-renders its heading has not changed what it is
    asking the user for, and re-matching on that would throw away the session on every
    navigation.

    Order matters. Two pages with the same fields in a different order are different
    pages, and the fill would land differently.

    Recomputed rather than trusted: a client-supplied hash means a stale plan can be
    replayed against a page that has since changed, which is the check `/plan` depends on.
    """
    path = urlsplit(url).path

    triples = _TRIPLE_SEPARATOR.join(
        _FIELD_SEPARATOR.join(
            (field.field_id, normalize_label(field.label), field.type.strip().casefold()),
        )
        for field in fields
    )

    payload = f"{path}{_TRIPLE_SEPARATOR}{triples}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def host_of(url: str) -> str:
    """
    The host, for a log line. Never the path, never the query.

    A government portal puts an application number in a query string, so the full URL is
    personal data. The host is enough to tell which portal a request came from.
    """
    return urlsplit(url).netloc or "unknown"
