"""
Drop the dedupe index on `documents`.

The exported corpus this table is now backfilled from (`scripts.load_documents`) turned out
to contain real, exact-content duplicates across several re-ingestion runs of the same
source — not the accidental double-insert `uq_documents_source_content` was written to catch,
but the corpus's own history, and the user's explicit choice was to load every one of those
rows rather than let the index silently swallow 1325 of them.

That is a real trade-off, not a free one: `uq_documents_source_content` is the same
constraint the `documents.py` model docstring credits with fixing the prototype's bug, where
one duplicated FAQ chunk filled every retrieval result slot. Dropping it removes that
backstop for `scripts.ingest` too, which still avoids the embedding spend via its own
Python-side hash check but no longer has the database's guarantee behind it if two ingestion
runs ever overlap. `src/documents/ingestion.py`'s `insert_documents` was written against this
index and is updated in the same change that adds this migration.

Revision ID: 0010_drop_documents_dedupe_index
Revises: 0009_results
Create Date: 2026-09-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010_drop_documents_dedupe_index"
down_revision: str | None = "0009_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEDUPE_INDEX = "uq_documents_source_content"


def upgrade() -> None:
    op.drop_index(DEDUPE_INDEX, table_name="documents")


def downgrade() -> None:
    """Reverse of upgrade.

    Recreating this index on a table that has since accepted duplicate rows will fail —
    exactly as it should: the index cannot be honest about uniqueness while the rows it
    would apply to are not unique. Deduplicate first (delete the rows this migration's
    upgrade let in) if a downgrade is ever actually needed.
    """
    op.create_index(
        DEDUPE_INDEX,
        "documents",
        ["source_url", "content_sha256"],
        unique=True,
    )
