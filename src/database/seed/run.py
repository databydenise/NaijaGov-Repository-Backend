"""
Load the reference data in `data/` into the database.

Idempotent: every write is an upsert on the primary key, so running this twice leaves
identical row counts. It never truncates, and it never touches `users`, `profiles`,
`extension_tokens`, `sessions`, or `action_log` — the tables that hold user data.

Runs on the direct connection (port 5432), like migrations.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.seed.exceptions import SeedValidationError
from src.database.seed.schemas import RuleSeed, WorkflowSeed, WorkflowStepSeed
from src.database.seed.validate import check_seed_data
from src.database.session import create_cli_engine
from src.knowledge.models import Rule
from src.workflows.models import Workflow, WorkflowStep

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

WORKFLOWS_FILE = DATA_DIR / "workflows.json"
STEPS_FILE = DATA_DIR / "steps.json"
RULES_FILE = DATA_DIR / "rules.json"


def _load(path: Path, adapter: TypeAdapter[Any]) -> Any:  # noqa: ANN401
    """Read one JSON file and validate it, naming the file if it fails."""
    if not path.exists():
        raise SeedValidationError(f"Missing seed file: {path.name}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    try:
        return adapter.validate_python(raw)
    except ValidationError as exc:
        raise SeedValidationError(f"{path.name} is invalid:\n{exc}") from exc


def load_seed_data() -> tuple[
    list[WorkflowSeed],
    list[WorkflowStepSeed],
    list[RuleSeed],
]:
    """Validate all three files, then check them against each other."""
    workflows = _load(WORKFLOWS_FILE, TypeAdapter(list[WorkflowSeed]))
    steps = _load(STEPS_FILE, TypeAdapter(list[WorkflowStepSeed]))
    rules = _load(RULES_FILE, TypeAdapter(list[RuleSeed]))

    check_seed_data(workflows, steps, rules)

    return workflows, steps, rules


async def _upsert(
    db: AsyncSession,
    model: type[Workflow] | type[WorkflowStep] | type[Rule],
    rows: list[dict[str, Any]],
) -> int:
    """Insert rows, updating every non-key column on conflict. Returns rows written."""
    if not rows:
        return 0

    statement = insert(model).values(rows)
    updatable = {
        column: statement.excluded[column] for column in rows[0] if column != "id"
    }
    statement = statement.on_conflict_do_update(
        index_elements=["id"],
        set_=updatable,
    )

    await db.execute(statement)

    return len(rows)


async def _delete_reference_data(db: AsyncSession) -> None:
    """Clear the reference tables only, child-first. User tables are never touched."""
    await db.execute(delete(Rule))
    await db.execute(delete(WorkflowStep))
    await db.execute(delete(Workflow))


async def seed(*, reset_reference: bool = False) -> dict[str, int]:
    """Validate the seed files and write them. Returns rows written per table."""
    workflows, steps, rules = load_seed_data()

    engine = create_cli_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as db:
            if reset_reference:
                await _delete_reference_data(db)

            written = {
                "workflows": await _upsert(
                    db,
                    Workflow,
                    [workflow.model_dump() for workflow in workflows],
                ),
                "workflow_steps": await _upsert(
                    db,
                    WorkflowStep,
                    [step.model_dump() for step in steps],
                ),
                "rules": await _upsert(
                    db,
                    Rule,
                    [
                        {**rule.model_dump(), "source_url": str(rule.source_url)}
                        for rule in rules
                    ],
                ),
            }

            await db.commit()
    finally:
        await engine.dispose()

    # The placeholder count is in the summary on purpose: it should look wrong once real
    # content lands, and nobody should have to query the database to notice.
    logger.info(
        "registry: %d workflow(s), %d step(s), %d rule(s) (%d placeholder)",
        written["workflows"],
        written["workflow_steps"],
        written["rules"],
        sum(1 for rule in rules if rule.is_placeholder),
    )

    return written
