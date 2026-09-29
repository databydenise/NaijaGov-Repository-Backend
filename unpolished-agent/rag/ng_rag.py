"""
ng_rag.py

Production-facing Nigeria Driver's Licence / FRSC RAG retrieval module.

The Colab notebook remains responsible for crawling/scraping, navigation
discovery, deduplication, chunking, embeddings, and Supabase ingestion.

This module is responsible only for retrieval for the backend AI agent.

Backend contract:
    search_government_information(query: str, limit: int = 3)
"""

import os
from typing import Any

import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")

SUPABASE_PORT = int(os.getenv("SUPABASE_PORT", "5432"))
SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")
SUPABASE_USER = os.getenv(
    "SUPABASE_USER",
    f"postgres.{SUPABASE_PROJECT_REF}" if SUPABASE_PROJECT_REF else "",
)

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def _validate_config() -> None:
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "SUPABASE_HOST": SUPABASE_HOST,
        "SUPABASE_DB_PASSWORD": SUPABASE_DB_PASSWORD,
        "SUPABASE_PROJECT_REF": SUPABASE_PROJECT_REF,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Clients / database
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _client

    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        _client = OpenAI(api_key=OPENAI_API_KEY)

    return _client


def _get_db_connection():
    _validate_config()

    conn = psycopg2.connect(
        host=SUPABASE_HOST,
        port=SUPABASE_PORT,
        database=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_DB_PASSWORD,
    )

    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _embed_query(query: str) -> list[float]:
    client = _get_openai_client()

    cleaned_query = query.replace("\n", " ").strip()

    response = client.embeddings.create(
        input=[cleaned_query],
        model=EMBEDDING_MODEL,
    )

    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Main RAG retrieval function
# ---------------------------------------------------------------------------


def search_government_information(
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Search the FRSC / Nigeria Driver's Licence knowledge base.

    Returns dictionaries containing:
        source
        content
        url
        relevance_score
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    query_embedding = _embed_query(query)

    conn = None

    try:
        conn = _get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    title,
                    context,
                    source_url,
                    (embedding <=> %s::vector) AS distance
                FROM documents
                WHERE embedding IS NOT NULL
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

    except Exception as exc:
        raise RuntimeError(
            f"Government information retrieval failed: {exc}"
        ) from exc

    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# OpenAI tool definition for the backend AI agent
# ---------------------------------------------------------------------------

GOVERNMENT_INFORMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "search_government_information",
        "description": (
            "Search the Nigeria Driver's Licence and FRSC knowledge base "
            "for official process information and verified website "
            "navigation instructions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The user's question or navigation request."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

GOV_TOOLS = [GOVERNMENT_INFORMATION_TOOL]


__all__ = [
    "search_government_information",
    "GOVERNMENT_INFORMATION_TOOL",
    "GOV_TOOLS",
]
