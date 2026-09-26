"""Async engine, session factory, and the FastAPI session dependency."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.config import settings

# Connections the app keeps open and reuses, so a request never pays to open one. Ten at
# full stretch, which is polite against a default `max_connections` of 100.
POOL_SIZE = 5
MAX_OVERFLOW = 5

# We connect straight to Postgres, so asyncpg's prepared-statement cache is left on — it
# is a real saving on repeated queries.
#
# If a transaction pooler (PgBouncer, Supavisor, RDS Proxy) is ever put in front of this
# database, that cache has to be disabled and prepared statements uniquely named, or
# asyncpg raises `prepared statement "__asyncpg_stmt_1__" already exists` once two
# requests share a backend. The fix is three connect_args — `statement_cache_size: 0`,
# `prepared_statement_cache_size: 0`, and a `prepared_statement_name_func` returning a
# uuid — and it only ever fails behind the pooler, never in local testing.
engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    echo=settings.db_echo,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


def create_cli_engine() -> AsyncEngine:
    """
    A throwaway engine for Alembic and the seed/cleanup commands.

    Unpooled: a command builds one, runs, and disposes of it, so nothing is left holding a
    connection after the process exits.
    """
    return create_async_engine(
        settings.async_database_url,
        echo=settings.db_echo,
        poolclass=NullPool,
    )


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Yield a session, committing on success and rolling back on any exception."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
