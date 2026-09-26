"""
Application settings.

Parsed once at import time. A missing or malformed value raises here, so the process
fails at boot rather than at the first request that needed the value.
"""

from functools import cached_property
from typing import Literal

from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.tokens.utils import TOKEN_PREFIX

ASYNCPG_SCHEME = "postgresql+asyncpg"

# A demo deployed with `secret` in the environment is the kind of thing that gets
# screenshotted, so this is a boot failure and not a warning.
MIN_JWT_SECRET_BYTES = 32

# `secrets.token_urlsafe(32)` produces 43 characters; anything under 32 is not random enough
# to stand in for a generated token.
MIN_DEMO_TOKEN_BODY = 32


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

    # --- Database ---
    database_url: PostgresDsn
    db_echo: bool = False

    # --- Deployment ---
    env: Literal["development", "staging", "production"] = "development"

    # Exact origins for CORS. Never `*`: a wildcard with credentials is both rejected by
    # browsers and a credential-theft amplifier.
    web_origin: str = "http://localhost:3000"

    # e.g. chrome-extension://<id>. Left unset until the extension is loaded, and omitted
    # from the CORS list while it is empty rather than defaulting to something permissive.
    extension_origin: str | None = None

    # --- Web session (the signed cookie the web app carries) ---
    jwt_secret: str
    session_ttl_days: int = 7

    # --- Extension tokens ---
    token_ttl_days: int = 30

    # --- Chat/fill sessions in the database. Not the web session above; the two are
    # different lifetimes on purpose and the names are the spec's. ---
    session_ttl_hours: int = 24

    # --- Demo mode ---
    demo_mode: bool = False
    demo_email: str = "demo@example.com"
    demo_password: str = "demo-account-not-for-real-use"

    # A fixed extension token for the demo account, installed at startup. Fixed rather than
    # generated because `fastapi dev` restarts on every file save, and a token that changed
    # on each restart would break the one written on the judging sheet. Optional: without
    # it, the demo account still exists and can issue a token through the web app.
    demo_token: str | None = None

    @field_validator("demo_token")
    @classmethod
    def _demo_token_well_formed(cls, value: str | None) -> str | None:
        if not value:
            return None

        body = value.removeprefix(TOKEN_PREFIX)

        # Same strength as a generated token. A demo instance can be reachable from the
        # internet, and `ngv_demo` would be guessed in minutes.
        if not value.startswith(TOKEN_PREFIX) or len(body) < MIN_DEMO_TOKEN_BODY:
            message = (
                f"DEMO_TOKEN must start with '{TOKEN_PREFIX}' followed by at least "
                f"{MIN_DEMO_TOKEN_BODY} random characters. Generate one with: python -c "
                "\"import secrets; print('ngv_' + secrets.token_urlsafe(32))\""
            )
            raise ValueError(message)

        return value

    @field_validator("jwt_secret")
    @classmethod
    def _secret_long_enough(cls, value: str) -> str:
        if len(value.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
            message = (
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes. "
                "Generate one with: python -c "
                "'import secrets; print(secrets.token_urlsafe(48))'"
            )
            raise ValueError(message)

        return value

    @cached_property
    def async_database_url(self) -> str:
        """The database URL on the asyncpg dialect."""
        return _as_asyncpg_url(self.database_url)

    @cached_property
    def cors_origins(self) -> list[str]:
        """Exactly the origins we serve, with no empty entries."""
        return [origin for origin in (self.web_origin, self.extension_origin) if origin]


def _refuse_demo_mode_in_production(loaded: Settings) -> None:
    """
    Fail the boot when the demo account would exist in production.

    Deliberately not a pydantic `model_validator`: a validation error renders the model's
    input, which would put part of `JWT_SECRET` in the crash output. This raises a plain
    error that names the two offending variables and nothing else.
    """
    if loaded.demo_mode and loaded.env == "production":
        message = (
            "DEMO_MODE=true with ENV=production. The demo account has a known password "
            "and a long-lived token; refusing to boot."
        )
        raise RuntimeError(message)


settings = Settings()  # type: ignore[call-arg]  # values come from the environment

_refuse_demo_mode_in_production(settings)
