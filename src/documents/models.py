"""Ingested chunks of official source material.

One row is one searchable chunk, not one document: a PDF page becomes several rows, and
each carries the URL it came from. `source_url` is NOT NULL for the same reason it is on
`rules` — a claim without a source is not evidence, and the constraint is what makes that
true in the data rather than only in a prompt.

Nothing in the API writes to this table. It is filled by `python -m scripts.ingest`, run
by hand, so that no request path can ever fetch from a government site.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Index, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.documents.constants import EMBEDDING_DIMENSIONS


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Narrowing a search to one agency's material, when the caller knows which.
        Index("ix_documents_agency_service", "agency", "service"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Human-readable provenance, shown beside a retrieved chunk: "FAQ: How do I renew…",
    # "Compendium 2025 (page 14, part 2)".
    title: Mapped[str] = mapped_column(Text, nullable=False)

    # "Federal Road Safety Corps (FRSC)" / "Driver's Licence". Both are filter columns, so
    # a search can be scoped when the workflow is already known.
    agency: Mapped[str] = mapped_column(Text, nullable=False)
    service: Mapped[str] = mapped_column(Text, nullable=False)

    # The chunk itself. This is the text the agent is allowed to ground an answer in, and
    # the text the guard checks a claim against.
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # SHA-256 of the whitespace-normalised chunk. `scripts.ingest` reads the hashes already
    # stored for a URL before embedding, so a re-run costs nothing for a chunk already here —
    # a Python-side check now, not a database constraint. Migration 0010 dropped the unique
    # index this used to double as half of, on request, to let in a corpus that already
    # contained exact-content duplicates across several ingestion runs.
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)

    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # faq | page | pdf. Not an enum: adding a kind should be a seed change, not a migration.
    source_kind: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )

    # Which model produced `embedding`. Per row rather than per deployment, so a model
    # change is visible as a mixed corpus instead of silently corrupting similarity.
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
