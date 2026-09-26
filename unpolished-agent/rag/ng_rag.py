"""
NG Government Information RAG retrieval module.

This module is the backend-facing part of the RAG system built for the
Nigeria government-services browser extension.

It does NOT scrape websites or populate the vector database. Those are
offline/data-ingestion jobs. The backend only needs this module to search
the existing Supabase pgvector database when the agent decides that it
needs government information.
"""

import os
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# These must be provided by the backend environment, NOT Colab userdata.
#
# Required:
#   OPENAI_API_KEY
#   SUPABASE_HOST
#   SUPABASE_DB_PASSWORD
#
# Optional:
#   SUPABASE_PROJECT_REF
#
# Example:
#   SUPABASE_PROJECT_REF=cpzayxaejvktgerjcmpx
#
PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "cpzayxaejvktgerjcmpx")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

if not SUPABASE_HOST:
    raise RuntimeError("Missing SUPABASE_HOST environment variable.")

if not SUPABASE_DB_PASSWORD:
    raise RuntimeError("Missing SUPABASE_DB_PASSWORD environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------
def _get_db_connection():
    """Create a connection to the existing Supabase PostgreSQL database."""
    return psycopg2.connect(
        host=SUPABASE_HOST,
        port=5432,
        database="postgres",
        user=f"postgres.{PROJECT_REF}",
        password=SUPABASE_DB_PASSWORD,
    )


# ---------------------------------------------------------------------------
# RAG retrieval tool
# ---------------------------------------------------------------------------
def search_government_information(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """
    Search the government-information vector database.

    The query is converted into an OpenAI embedding and matched against
    the existing pgvector embeddings in the Supabase `documents` table.

    Args:
        query: Natural-language question or search phrase.
        limit: Maximum number of matching document chunks to return.

    Returns:
        A list of dictionaries containing the source title, retrieved
        content, source URL, and cosine-similarity relevance score.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    limit = max(1, min(int(limit), 10))

    response = client.embeddings.create(
        input=[query.replace("\n", " ")],
        model="text-embedding-3-small",
    )
    query_embedding = response.data[0].embedding

    conn = _get_db_connection()

    try:
        register_vector(conn)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                title,
                context,
                source_url,
                (embedding <=> %s::vector) AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT %s;
            """,
            (query_embedding, limit),
        )

        rows = cursor.fetchall()

        return [
            {
                "source": title,
                "content": context,
                "url": source_url,
                "relevance_score": float(1 - distance),
            }
            for title, context, source_url, distance in rows
        ]

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# OpenAI function-tool definition
# ---------------------------------------------------------------------------
GOVERNMENT_INFORMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "search_government_information",
        "description": (
            "Search the official Nigeria government-information database "
            "for FAQs, procedures, portal guidance, regulations, and other "
            "government-service information. Use this when the user asks "
            "about government processes or information that should be "
            "verified against the retrieved sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The specific question, search phrase, or keywords "
                        "to look up in the government information database."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of matching document snippets to retrieve.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

# Convenient form for passing the tool list directly to an OpenAI request.
GOV_TOOLS = [GOVERNMENT_INFORMATION_TOOL]


__all__ = [
    "search_government_information",
    "GOVERNMENT_INFORMATION_TOOL",
    "GOV_TOOLS",
]
