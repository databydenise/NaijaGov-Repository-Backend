"""
ng_session_profile.py

Session/profile/message persistence layer for the Nigerian bureaucracy copilot.

This is the backend-callable version of the database work from the Colab notebook.
It deliberately does NOT contain Colab-specific code (drive.mount/userdata) or
connection test prints.

Environment variables required:
    SUPABASE_HOST
    SUPABASE_DB_PASSWORD

Optional:
    SUPABASE_PROJECT_REF  (defaults to the project ref used in the original notebook)
    SUPABASE_DB_PORT      (defaults to 5432)
    SUPABASE_DB_NAME      (defaults to "postgres")

Dependency:
    pip install psycopg2-binary
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


DEFAULT_PROJECT_REF = "cpzayxaejvktgerjcmpx"

PROFILE_FIELDS = {
    "name",
    "email",
    "phone",
    "address",
    "state",
    "lga",
}


def _get_connection():
    """Create a PostgreSQL connection using environment variables."""
    host = os.getenv("SUPABASE_HOST")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    project_ref = os.getenv("SUPABASE_PROJECT_REF", DEFAULT_PROJECT_REF)

    if not host:
        raise RuntimeError("SUPABASE_HOST is not set.")
    if not password:
        raise RuntimeError("SUPABASE_DB_PASSWORD is not set.")

    return psycopg2.connect(
        host=host,
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=f"postgres.{project_ref}",
        password=password,
    )


@contextmanager
def get_connection():
    """Yield a database connection and automatically close it."""
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the sessions, profiles and messages tables if they do not exist.

    This is a setup/migration helper. It is NOT run automatically when this
    module is imported.
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE EXTENSION IF NOT EXISTS pgcrypto;

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    session_id UUID PRIMARY KEY
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    address TEXT,
                    state TEXT,
                    lga TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id UUID NOT NULL
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        conn.commit()


def create_session() -> str:
    """Create a new session and return its UUID as a string."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO sessions DEFAULT VALUES
                RETURNING session_id;
                """
            )
            session_id = cursor.fetchone()[0]
        conn.commit()

    return str(session_id)


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Return an existing session ID, or create a new session when none is supplied.

    This is the simplest entry point for the backend at the start of a request:
        session_id = get_or_create_session(session_id)
    """
    if session_id:
        if session_exists(session_id):
            return session_id
        raise ValueError(f"Session {session_id} does not exist.")

    return create_session()


def session_exists(session_id: str) -> bool:
    """Return True if the session exists."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM sessions WHERE session_id = %s
                );
                """,
                (session_id,),
            )
            return bool(cursor.fetchone()[0])


def get_profile(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the profile attached to a session, or None if it does not exist."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT session_id, name, email, phone, address, state, lga
                FROM profiles
                WHERE session_id = %s;
                """,
                (session_id,),
            )
            row = cursor.fetchone()

    return dict(row) if row else None


def upsert_profile(
    session_id: str,
    *,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None,
    state: Optional[str] = None,
    lga: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or update the profile for a session.

    Only fields explicitly passed to this function are changed. Existing
    values are preserved for omitted fields.
    """
    if not session_exists(session_id):
        raise ValueError(f"Session {session_id} does not exist.")

    values = {
        "name": name,
        "email": email,
        "phone": phone,
        "address": address,
        "state": state,
        "lga": lga,
    }

    # Keep only values that the caller actually supplied.
    supplied = {k: v for k, v in values.items() if v is not None}

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Ensure a profile row exists.
            cursor.execute(
                """
                INSERT INTO profiles (session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING;
                """,
                (session_id,),
            )

            if supplied:
                assignments = ", ".join(
                    f"{field} = %s" for field in supplied
                )
                params = list(supplied.values()) + [session_id]

                cursor.execute(
                    f"""
                    UPDATE profiles
                    SET {assignments}
                    WHERE session_id = %s;
                    """,
                    params,
                )

        conn.commit()

    return get_profile(session_id)  # type: ignore[return-value]


def add_message(session_id: str, role: str, content: str) -> str:
    """
    Store a conversation message.

    role is normally something like: "user", "assistant", or "system".
    Returns the new message UUID as a string.
    """
    if not session_exists(session_id):
        raise ValueError(f"Session {session_id} does not exist.")

    if not content.strip():
        raise ValueError("Message content cannot be empty.")

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (%s, %s, %s)
                RETURNING message_id;
                """,
                (session_id, role, content),
            )
            message_id = cursor.fetchone()[0]
        conn.commit()

    return str(message_id)


def get_messages(
    session_id: str,
    *,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return the most recent messages for a session, oldest-to-newest."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT message_id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()

    # The query gets newest first; return chronological order to the caller.
    return [dict(row) for row in reversed(rows)]


def save_interaction(
    session_id: str,
    *,
    user_message: Optional[str] = None,
    assistant_message: Optional[str] = None,
    profile_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save one copilot interaction in one call.

    Optionally stores:
      - a user message
      - an assistant message
      - profile updates

    Returns the updated session context.
    """
    if not session_exists(session_id):
        raise ValueError(f"Session {session_id} does not exist.")

    if profile_updates:
        invalid_fields = set(profile_updates) - PROFILE_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Unknown profile fields: {sorted(invalid_fields)}"
            )

        upsert_profile(session_id, **profile_updates)

    if user_message is not None:
        add_message(session_id, "user", user_message)

    if assistant_message is not None:
        add_message(session_id, "assistant", assistant_message)

    return get_session_context(session_id)


def get_session_context(
    session_id: str,
    *,
    message_limit: int = 20,
) -> Dict[str, Any]:
    """
    Return the session's profile + recent messages in one backend-friendly object.

    This is the main convenience function for the copilot/orchestrator.
    """
    if not session_exists(session_id):
        raise ValueError(f"Session {session_id} does not exist.")

    return {
        "session_id": session_id,
        "profile": get_profile(session_id),
        "messages": get_messages(session_id, limit=message_limit),
    }
