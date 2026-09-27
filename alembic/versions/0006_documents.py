"""
The documents table: ingested official source material, searchable by vector.

Requires the `vector` extension (pgvector) on the server. On a managed host that is a
setting; on a self-managed one it is a package install. `CREATE EXTENSION` below fails
loudly rather than skipping the table, because a migration that half-runs is worse than
one that stops.

Revision ID: 0006_documents
Revises: 0005_sessions_tab_id
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0006_documents"
down_revision: str | None = "0005_sessions_tab_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# text-embedding-3-small. Mirrors EMBEDDING_DIMENSIONS in src/documents/constants.py;
# duplicated rather than imported, because a migration must keep describing the schema it
# created even after the application constant moves on.
EMBEDDING_DIMENSIONS = 1536

DEDUPE_INDEX = "uq_documents_source_content"
AGENCY_INDEX = "ix_documents_agency_service"
EMBEDDING_INDEX = "ix_documents_embedding"

# ivfflat partitions the vectors into this many lists. The usual guidance is rows/1000 for
# a small corpus; at ~300 rows the index is not doing real work yet, and 100 is the value
# to leave in place as the corpus grows. Building it now rather than later is deliberate:
# adding it to a full table is slow and easy to forget.
IVFFLAT_LISTS = 100


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("agency", sa.Text(), nullable=False),
        sa.Column("service", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # SHA-256 of the whitespace-normalised chunk. Half the dedupe key.
        sa.Column("content_sha256", sa.Text(), nullable=False),
        # NOT NULL for the same reason it is on `rules`: evidence without a source is not
        # evidence, and every answer built on this row shows the URL.
        sa.Column("source_url", sa.Text(), nullable=False),
        # faq | page | pdf. Text rather than an enum, so adding a kind is a manifest change
        # rather than a migration.
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        # Per row, so a model change shows up as a mixed corpus instead of quietly
        # comparing vectors that were never comparable.
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
    )

    # The constraint that makes the prototype's duplicate-ingestion bug unrepeatable: it
    # inserted the FAQ set twice, and one duplicated chunk then filled every result slot.
    op.create_index(
        DEDUPE_INDEX,
        "documents",
        ["source_url", "content_sha256"],
        unique=True,
    )

    op.create_index(AGENCY_INDEX, "documents", ["agency", "service"])

    op.execute(
        f"CREATE INDEX {EMBEDDING_INDEX} ON documents "
        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {IVFFLAT_LISTS})",
    )

    # Defence in depth, as on every other table in this schema. No policy is written, so a
    # future read-only role sees zero rows until someone writes one deliberately. The
    # application connects as the owner, which RLS does not restrict by default.
    op.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    """Reverse of upgrade. A migration that cannot be reversed cannot be tested twice.

    The `vector` extension is left installed: other objects may come to depend on it, and
    dropping an extension takes its types with it.
    """
    op.drop_index(EMBEDDING_INDEX, table_name="documents")
    op.drop_index(AGENCY_INDEX, table_name="documents")
    op.drop_index(DEDUPE_INDEX, table_name="documents")
    op.drop_table("documents")
