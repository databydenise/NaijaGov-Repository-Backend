"""
Application settings.

Parsed once at import time. A missing or malformed value raises here, so the process
fails at boot rather than at the first request that needed the value.
"""

from functools import cached_property

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

ASYNCPG_SCHEME = "postgresql+asyncpg"


def _as_asyncpg_url(dsn: PostgresDsn) -> str:
    """
    Rewrite a libpq DSN to the SQLAlchemy asyncpg dialect.

    `postgresql://` is what Postgres tooling hands out; SQLAlchemy needs the driver named
    explicitly or it reaches for psycopg2, which is not a dependency of this project.
    """
    url = str(dsn)
    _, _, remainder = url.partition("://")

    return f"{ASYNCPG_SCHEME}://{remainder}"


class Settings(BaseSettings):
    """
    Environment-backed configuration.

    One connection string: the app, Alembic, and the CLIs all talk to the same Postgres
    database directly, with no connection pooler in between.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: PostgresDsn
    db_echo: bool = False
    session_ttl_hours: int = 24

    @cached_property
    def async_database_url(self) -> str:
        """The database URL on the asyncpg dialect."""
        return _as_asyncpg_url(self.database_url)


settings = Settings()  # type: ignore[call-arg]  # values come from the environment
