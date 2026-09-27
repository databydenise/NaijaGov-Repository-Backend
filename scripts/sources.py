"""
Fetching and text extraction for ingestion.

The only module in the repository that makes outbound requests to a government website.
Nothing under `src/` imports it, so no request path can reach one.

Politeness is not optional here. These are public services run on public money, often on
modest infrastructure: one request at a time, a delay between them, an honest User-Agent
that says who is asking, and `robots.txt` respected even where it costs us a source.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.documents.constants import SourceKind
from src.documents.utils import (
    Chunk,
    faq_chunks,
    normalize_whitespace,
    page_chunks,
    pdf_page_chunks,
)

logger = logging.getLogger(__name__)

# Honest, and traceable back to this project. The prototype sent a copied Chrome string,
# which tells an administrator watching their logs nothing about who is fetching or why.
USER_AGENT = (
    "NaijaGovBot/0.1 (+https://github.com/naijagov; non-commercial; "
    "ingests public guidance for a citizen assistance tool)"
)

REQUEST_TIMEOUT_SECONDS = 30.0

# A robots.txt that 404s or errors is treated as no robots.txt at all, which is the
# convention. Anything below 400 is a body worth parsing.
HTTP_ERROR_STATUS = 400

# Between requests, so a run never looks like a burst to the host.
POLITE_DELAY_SECONDS = 2.0

# A compendium PDF is a few megabytes; anything far past that is not a document we meant
# to fetch, and streaming it into memory would be the first thing to go wrong.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

# Text shorter than this on a PDF page is a page number or a running header.
MIN_PDF_PAGE_CHARS = 50

# Tags whose text is navigation and boilerplate rather than guidance.
_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "form")


@dataclass(frozen=True)
class Source:
    """One entry from `data/sources.yaml`."""

    url: str
    kind: str
    agency: str
    service: str
    title: str | None = None

    @property
    def display_title(self) -> str:
        """The stem for this source's chunk titles."""
        if self.title:
            return self.title

        path = urlparse(self.url).path.rstrip("/")

        return path.rsplit("/", 1)[-1] or urlparse(self.url).netloc


class SourceError(RuntimeError):
    """A source could not be fetched or parsed. Reported per source; the run continues."""


def load_manifest(path: str) -> list[Source]:
    """Read and validate `sources.yaml`.

    Every field is required except `title`, and an unknown `kind` is refused rather than
    guessed: a typo silently extracting nothing would look like a site that changed.
    """
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(entries, list) or not entries:
        message = f"{path} must hold a non-empty list of sources"
        raise SourceError(message)

    sources = []

    for index, entry in enumerate(entries, start=1):
        missing = [key for key in ("url", "kind", "agency", "service") if not entry.get(key)]

        if missing:
            message = f"{path} entry {index} is missing: {', '.join(missing)}"
            raise SourceError(message)

        if entry["kind"] not in SourceKind.ALL:
            message = (
                f"{path} entry {index} has kind '{entry['kind']}'; "
                f"expected one of {', '.join(SourceKind.ALL)}"
            )
            raise SourceError(message)

        sources.append(
            Source(
                url=entry["url"],
                kind=entry["kind"],
                agency=entry["agency"],
                service=entry["service"],
                title=entry.get("title"),
            ),
        )

    return sources


# One robots.txt per host per run. Several sources share a host, and re-fetching the same
# file for each of them is exactly the impoliteness this module is trying to avoid.
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _robots_for(client: httpx.Client, url: str) -> urllib.robotparser.RobotFileParser | None:
    """The parsed robots.txt for this URL's host, or None when there is nothing to obey."""
    host = urlparse(url).netloc

    if host in _robots_cache:
        return _robots_cache[host]

    robots_url = urljoin(url, "/robots.txt")
    parser: urllib.robotparser.RobotFileParser | None = None

    try:
        response = client.get(robots_url, timeout=10.0)
    except httpx.HTTPError as exc:
        logger.info("No robots.txt read for %s (%s); proceeding", robots_url, type(exc).__name__)
    else:
        if response.status_code < HTTP_ERROR_STATUS:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())

    _robots_cache[host] = parser

    return parser


def robots_allows(client: httpx.Client, url: str) -> bool:
    """Whether `robots.txt` permits our User-Agent to fetch this URL.

    A `robots.txt` that cannot be read is treated as permission. That is the convention,
    and the alternative — refusing every source whenever a host 500s on one file — makes
    ingestion fail for a reason that has nothing to do with the source.
    """
    parser = _robots_for(client, url)

    if parser is None:
        return True

    return parser.can_fetch(USER_AGENT, url)


def build_client() -> httpx.Client:
    """A client that identifies itself and follows redirects."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def fetch(client: httpx.Client, url: str) -> httpx.Response:
    """GET one URL, politely, refusing a response too large to be a document."""
    time.sleep(POLITE_DELAY_SECONDS)

    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        message = f"could not fetch {url}: {type(exc).__name__}"
        raise SourceError(message) from exc

    if len(response.content) > MAX_DOWNLOAD_BYTES:
        message = f"{url} returned {len(response.content)} bytes, over the download cap"
        raise SourceError(message)

    return response


# FAQ layouts, in the order they are tried: (container class, question class, answer class).
# Bootstrap panels first, then the older accordion markup. `<details>` is handled separately
# because its question and answer share one element.
_FAQ_LAYOUTS = (
    ("panel", "panel-heading", "panel-collapse"),
    ("accordion-group", "accordion-heading", "accordion-body"),
)


def _pairs_from_classes(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Question/answer pairs out of a panel or accordion layout, whichever matches first."""
    for container, question_class, answer_class in _FAQ_LAYOUTS:
        pairs = []

        for panel in soup.find_all(class_=container):
            heading = panel.find(class_=question_class)
            body = panel.find(class_=answer_class)

            if heading and body:
                pairs.append((heading.get_text(" ", strip=True), body.get_text(" ", strip=True)))

        if pairs:
            return pairs

    return []


def _pairs_from_details(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Question/answer pairs out of `<details><summary>` markup.

    The summary is removed from the element before the answer is read, or the question
    would appear twice in the text that gets embedded.
    """
    pairs = []

    for details in soup.find_all("details"):
        summary = details.find("summary")

        if not summary:
            continue

        question = summary.get_text(" ", strip=True)
        summary.extract()
        pairs.append((question, details.get_text(" ", strip=True)))

    return pairs


def _deduplicate_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop empty and repeated pairs, comparing on normalised text.

    The prototype's own duplicate check, kept: these pages render the same panel twice,
    once in a mobile block and once in a desktop one, and both land in the same soup. Not
    catching that is how one chunk ended up filling every result slot.
    """
    unique = []
    seen: set[tuple[str, str]] = set()

    for question, answer in pairs:
        key = (normalize_whitespace(question), normalize_whitespace(answer))

        if not all(key) or key in seen:
            continue

        seen.add(key)
        unique.append((question, answer))

    return unique


def extract_faq(html: str, source: Source) -> list[Chunk]:
    """Question and answer pairs out of an FAQ page.

    Three layouts are tried in turn: Bootstrap panels, the older accordion markup, and
    finally `<details>`/`<summary>`. A page that renders its FAQ with JavaScript yields
    nothing, and nothing is the right answer — a half-scraped FAQ is worse than none, and
    the exception says the layout may have changed rather than letting the run look clean.
    """
    soup = BeautifulSoup(html, "html.parser")
    pairs = _deduplicate_pairs(_pairs_from_classes(soup) or _pairs_from_details(soup))

    chunks: list[Chunk] = []

    for question, answer in pairs:
        chunks.extend(faq_chunks(question, answer))

    if not chunks:
        message = f"no FAQ pairs found at {source.url}; the page layout may have changed"
        raise SourceError(message)

    return chunks


def extract_page(html: str, source: Source) -> list[Chunk]:
    """Readable text out of a content page.

    Navigation, scripts and forms are removed and the remaining text is taken whole, rather
    than the prototype's keyword filter over every `div`. That filter both missed guidance
    phrased without its keywords and kept the same paragraph several times over, once per
    nesting level.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    text = normalize_whitespace(main.get_text(" ", strip=True))

    if not text:
        message = f"no readable text at {source.url}"
        raise SourceError(message)

    title = source.title or (soup.title.get_text(strip=True) if soup.title else source.display_title)

    return page_chunks(title, text)


def extract_pdf(content: bytes, source: Source) -> list[Chunk]:
    """Text per page of a PDF, page numbers preserved in the titles.

    A page whose extraction comes back near-empty is skipped rather than stored: scanned
    pages produce a handful of ligature noise, which embeds to nothing useful and dilutes
    the corpus.
    """
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:  # pypdf raises several unrelated types
        message = f"could not read {source.url} as a PDF: {type(exc).__name__}"
        raise SourceError(message) from exc

    chunks: list[Chunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001  # one bad page must not stop the file
            logger.warning("Page %d of %s failed to extract (%s)", page_number, source.url, type(exc).__name__)

            continue

        if len(text.strip()) < MIN_PDF_PAGE_CHARS:
            continue

        chunks.extend(pdf_page_chunks(source.display_title, page_number, text))

    if not chunks:
        message = f"no extractable text in {source.url}; it may be a scanned document"
        raise SourceError(message)

    return chunks


def collect_chunks(client: httpx.Client, source: Source) -> list[Chunk]:
    """Fetch one source and turn it into chunks, by kind."""
    if not robots_allows(client, source.url):
        message = f"robots.txt disallows {source.url}"
        raise SourceError(message)

    response = fetch(client, source.url)

    if source.kind == SourceKind.PDF:
        return extract_pdf(response.content, source)
    if source.kind == SourceKind.FAQ:
        return extract_faq(response.text, source)

    return extract_page(response.text, source)


def manifest_summary(sources: list[Source]) -> dict[str, Any]:
    """Counts by kind, for the run's opening line."""
    return {kind: sum(1 for source in sources if source.kind == kind) for kind in SourceKind.ALL}
