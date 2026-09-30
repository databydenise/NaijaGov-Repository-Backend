# NaijaGov-Repository
Backend services for NaijaGov

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (dependency manager — a `uv.lock` is checked in)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension available
  (e.g. [Postgres.app](https://postgresapp.com/) locally)

### 1. Install dependencies

```bash
uv sync
```

This creates `.venv` and installs everything pinned in `uv.lock`.

### 2. Configure environment

```bash
cp .env.example .env
```

Then edit `.env`. At minimum:

- `DATABASE_URL` — a Postgres database, e.g. after `createdb naijagov_dev`:
  `postgresql://your-macos-user@localhost:5432/naijagov_dev`
- `JWT_SECRET` — at least 32 bytes, generate with:
  `python -c 'import secrets; print(secrets.token_urlsafe(48))'`

Every other setting in `.env.example` has a comment explaining it and a safe default
(retrieval, the demo account, and the OpenAI key are all optional — the app boots without
them and degrades gracefully).

### 3. Set up the database

```bash
uv run alembic upgrade head
uv run python -m src.database.seed
```

Re-run the seed command any time; it's idempotent. Pass `--reset-reference` after editing a
seed file to reload workflows/steps/rules from scratch.

### 4. Start the server

```bash
uv run fastapi dev src/main.py
```

Visit `http://localhost:8000/health` to confirm it's up.

### Optional: demo account

Set `DEMO_MODE=true` in `.env` (refuses to start if `ENV=production`) and the server creates
a demo user and extension token on startup — see `src/demo/service.py`.

### Optional: retrieval / AI

Set `OPENAI_API_KEY` to enable vector search and model calls. Without it, `/health` reports
retrieval as `unconfigured` and searches return no results rather than failing.
