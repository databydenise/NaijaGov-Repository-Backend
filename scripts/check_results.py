"""
Check `POST /results` without a database or pytest.

    python -m scripts.check_results

Everything runs over the real app, through the real middleware and the real exception handlers, so
what these checks read is the body a caller would actually receive, wrapped shape and all. Only
the database is replaced — by a stub that **compiles every statement against the Postgres dialect**
and answers from dicts, so a column renamed in a model and not in the service fails here rather
than on the first request against a real database.

What it covers, against the spec's own definition of done and its error table:

 1. a run is recorded once, with the statuses it reported and the hint it earns
 2. the action type and the field come from the plan, never from the request
 3. an action id the plan does not contain is skipped rather than invented
 4. reporting the same run twice writes once and returns the same answer
 5. a plan claimed by a concurrent report replays that report, and writes nothing
 6. an unknown session, an unknown plan and another user's session are all `200`, not errors
 7. a plan reported against the wrong session is refused the same way
 8. a checkpoint records an event, normalises its kind for the counters, and outranks everything
 9. failures outrank finality; finality outranks nothing left to fix
10. the session's counters are incremented in SQL, not read-modify-written
11. the durable counters upsert on workflow, step, metric and day
12. the value / text / label tripwire, naming the key and echoing nothing
13. an unknown reason code, a missing reason, a bad status, and over-long input are all 400
14. the rate limit holds
15. no value, label or reason prose in any log line
16. the status vocabulary agrees across the Literal, the constant and the column CHECK
17. `/health`'s results block returns rows, and degrades to null rather than failing

What it cannot cover, and has to be done by hand: migration `0009`, the real primary key behind
the claim, the real unique index behind the counter upsert, and a genuine concurrent report.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Insert, Select, Update
from sqlalchemy.dialects import postgresql

from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.logging import RedactingFilter, configure_logging
from src.main import app
from src.plan import store as plan_store
from src.plan.schemas import PlannedActionOut
from src.rate_limit import reset_rate_limits
from src.results import health as results_health
from src.results.constants import (
    MAX_RESULTS,
    NEXT_HINTS,
    RATE_LIMIT,
)
from src.results.schemas import RunTotals, statuses_are_in_sync
from src.sessions.constants import ACTION_STATUSES
from src.sessions.models import Session
from src.tokens.dependencies import require_token
from src.workflows import service as workflows_service
from src.workflows.schemas import Step

# --- The run, the user, and the plan every case reports against --------------------------------

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_USER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
TOKEN_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
OTHER_SESSION_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")
PLAN_ID = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"

# Planted strings. None of these may appear in any log line.
PLANTED_LABEL = "Local Government Area SECRETLABEL77777"

STEPS = [
    Step(
        id="demo_reg_applicant",
        workflow_id="demo_reg",
        name="Applicant Information",
        index=1,
        field_labels=("Full Name", "Email Address", "Phone Number", "State"),
        is_final=False,
    ),
    Step(
        id="demo_reg_verify",
        workflow_id="demo_reg",
        name="Verification",
        index=2,
        field_labels=("Password", "One-Time Code", "Declaration"),
        is_final=True,
    ),
]


def approved_actions() -> tuple[PlannedActionOut, ...]:
    """
    The plan's approved actions, as `/plan` would have stored them.

    Labels and values are present here because a real `PendingPlan` carries them — which is part of
    what these checks are for: none of it may reach a row, a response or a log line.
    """
    return (
        PlannedActionOut(
            action_id="a1",
            type="fill",
            field_id="f1",
            label="Full Name",
            value="Adaeze Okonkwo",
            source="Your profile",
            source_ref="profile.full_name",
        ),
        PlannedActionOut(
            action_id="a2",
            type="fill",
            field_id="f2",
            label="Email Address",
            value="adaeze@example.com",
            source="Your profile",
            source_ref="profile.email",
        ),
        PlannedActionOut(
            action_id="a3",
            type="select",
            field_id="f4",
            label="State",
            value="Lagos",
            source="Your profile",
            source_ref="profile.state",
        ),
        PlannedActionOut(
            action_id="a4",
            type="fill",
            field_id="f5",
            label=PLANTED_LABEL,
            value="Ikeja",
            source="You told me just now",
            source_ref="chat.lga",
        ),
        PlannedActionOut(action_id="a5", type="highlight", field_id="f6", label="Password"),
    )


def results_body(
    *,
    session_id: uuid.UUID = SESSION_ID,
    plan_id: str = PLAN_ID,
    entries: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A well-formed report, with room to break one thing at a time."""
    return {
        "session_id": str(session_id),
        "plan_id": plan_id,
        "results": entries
        if entries is not None
        else [
            {"action_id": "a1", "field_id": "f1", "status": "ok"},
            {"action_id": "a2", "field_id": "f2", "status": "changed", "reason": "NOT_ACCEPTED"},
            {"action_id": "a3", "field_id": "f4", "status": "ok"},
            {"action_id": "a4", "field_id": "f5", "status": "ok"},
            {
                "action_id": "a5",
                "field_id": "f6",
                "status": "rejected",
                "reason": "SENSITIVE_FIELD",
            },
        ],
        "elapsed_ms": 1100,
        **extra,
    }


# --- The stub database ------------------------------------------------------------------------


@dataclass
class StubResult:
    """What `execute` returns: one scalar, or none."""

    value: Any = None

    def scalar_one_or_none(self) -> Any:  # noqa: ANN401  # a row, a plan id, or None
        return self.value


@dataclass
class StubDb:
    """
    Every statement `/results` issues, compiled for Postgres and answered from dicts.

    Faithful where it matters:

    - the session SELECT answers only when the statement's ids match the row, so "another user's
      session" is a check of the real WHERE clause rather than of this stub's opinion;
    - the `plan_reports` INSERT honours its conflict clause, so idempotency is exercised rather
      than assumed;
    - every statement is compiled, so a column name that does not exist fails here.
    """

    session: Session | None
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_rows: list[Any] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)
    # The parameters of every counter upsert. A multi-row INSERT carries its values as
    # `metric_m0`, `metric_m1`, … so the metric names are in here rather than in the SQL text.
    counter_params: list[dict[str, Any]] = field(default_factory=list)
    flushes: int = 0
    # Set to simulate another request claiming the plan between our read and our insert.
    steal_claim: bool = False

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> StubResult:
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = " ".join(str(compiled).split())
        self.statements.append(sql)
        params = compiled.params

        if isinstance(statement, Select):
            return self._select(sql, params)

        if isinstance(statement, Insert):
            return self._insert(sql, params)

        if isinstance(statement, Update):
            return StubResult(None)

        return StubResult(None)

    def _select(self, sql: str, params: dict[str, Any]) -> StubResult:
        if "FROM plan_reports" in sql:
            return StubResult(self.reports.get(params.get("plan_id_1")))

        if "FROM sessions" in sql:
            if self.session is None or not _matches(params, self.session):
                return StubResult(None)

            return StubResult(self.session)

        return StubResult(None)

    def _insert(self, sql: str, params: dict[str, Any]) -> StubResult:
        if "INTO results_counters" in sql:
            self.counter_params.append(params)

            return StubResult(None)

        if "INTO plan_reports" in sql:
            plan_id = params["plan_id"]

            if self.steal_claim:
                # Another request got there first. Its answer is now the stored one.
                self.reports[plan_id] = params["response"]

                return StubResult(None)

            if plan_id in self.reports:
                return StubResult(None)

            self.reports[plan_id] = params["response"]

            return StubResult(plan_id)

        return StubResult(None)

    async def flush(self) -> None:
        self.flushes += 1

    def add(self, row: Any) -> None:
        self.events.append(row)

    def add_all(self, rows: Sequence[Any]) -> None:
        self.action_rows.extend(rows)

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    @property
    def counter_statements(self) -> list[str]:
        return [sql for sql in self.statements if "INTO results_counters" in sql]

    @property
    def counter_metrics(self) -> set[str]:
        """Every metric name written to `results_counters`, across all upserts."""
        return {
            value
            for params in self.counter_params
            for key, value in params.items()
            if key.startswith("metric") and isinstance(value, str)
        }

    @property
    def counter_scopes(self) -> set[tuple[str, str]]:
        """Every `(workflow, step)` key the counters were written under."""
        scopes: set[tuple[str, str]] = set()

        for params in self.counter_params:
            suffixes = [key.removeprefix("metric") for key in params if key.startswith("metric")]
            scopes.update(
                (str(params[f"workflow_id{suffix}"]), str(params[f"step_id{suffix}"]))
                for suffix in suffixes
            )

        return scopes

    @property
    def session_updates(self) -> list[str]:
        return [sql for sql in self.statements if sql.upper().startswith("UPDATE SESSIONS")]


def _matches(params: dict[str, Any], session: Session) -> bool:
    """Whether the session row is the one a compiled SELECT is asking for."""
    bound = [value for value in params.values() if isinstance(value, uuid.UUID)]

    return all(value in (session.id, session.user_id) for value in bound)


def make_session(
    *,
    user_id: uuid.UUID = USER_ID,
    session_id: uuid.UUID = SESSION_ID,
    step_id: str | None = "demo_reg_applicant",
    workflow_id: str | None = "demo_reg",
) -> Session:
    """An unsaved session row, as the database would have handed it back."""
    session = Session(
        user_id=user_id,
        tab_id=1,
        workflow_id=workflow_id,
        step_id=step_id,
        page_hash="hash",
        cached_rule_ids=[],
        history=[],
        chat_values={},
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    session.id = session_id

    return session


# --- Harness ---------------------------------------------------------------------------------


class LogCapture(logging.Handler):
    """Every log line emitted during a case, with the app's own redaction applied."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.addFilter(RedactingFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


_DEFAULT_SESSION = object()


def build_harness(
    *,
    session: Any = _DEFAULT_SESSION,
    plan_session_id: uuid.UUID | None = SESSION_ID,
    plan_user_id: uuid.UUID = USER_ID,
    remember: bool = True,
) -> tuple[TestClient, StubDb, LogCapture]:
    """
    The app with its database replaced, a pending plan in the store, and a client to call it with.

    Global state — the rate limiter, both plan stores, the registry cache — is reset here, so a
    case cannot pass or fail because of the one before it.
    """
    reset_rate_limits()
    plan_store.reset_plans()

    db = StubDb(session=make_session() if session is _DEFAULT_SESSION else session)
    logs = LogCapture()

    if remember and plan_session_id is not None:
        plan_store.remember_plan(
            PLAN_ID,
            session_id=plan_session_id,
            user_id=plan_user_id,
            actions=approved_actions(),
        )

    async def stub_steps(*_args: Any, **_kwargs: Any) -> list[Step]:
        return STEPS

    workflows_service.get_steps = stub_steps  # type: ignore[assignment]

    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_token] = lambda: AuthedUser(
        id=USER_ID,
        email="demo@example.com",
        token_id=TOKEN_ID,
    )

    logging.getLogger().addHandler(logs)

    return TestClient(app, raise_server_exceptions=False), db, logs


def teardown(logs: LogCapture) -> None:
    logging.getLogger().removeHandler(logs)
    app.dependency_overrides.clear()


def expect(failures: list[str], condition: bool, description: str) -> None:  # noqa: FBT001
    if not condition:
        failures.append(description)


def data_of(response: Any) -> dict[str, Any]:  # noqa: ANN401  # httpx Response
    return response.json().get("data", {})


def error_of(response: Any) -> dict[str, Any]:  # noqa: ANN401  # httpx Response
    return response.json().get("error", {})


# --- Cases -----------------------------------------------------------------------------------


async def check_records_a_run() -> list[str]:
    """A whole run recorded once: the counts it reported, and the rows the plan describes."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        response = client.post("/results", json=results_body())
        data = data_of(response)
        run = data.get("run", {})

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(failures, data.get("acknowledged") is True, "a live plan was not acknowledged")
        expect(failures, data.get("recorded") == 5, f"recorded was {data.get('recorded')}")
        expect(failures, run.get("ok") == 3, f"ok was {run.get('ok')}")
        expect(failures, run.get("changed") == 1, f"changed was {run.get('changed')}")
        expect(failures, run.get("rejected") == 1, f"rejected was {run.get('rejected')}")
        expect(failures, run.get("failed") == 0, "nothing failed in this run")
        expect(
            failures,
            data.get("next_hint", {}).get("code") == "FIX_FIRST",
            f"hint was {data.get('next_hint', {}).get('code')} (a rejected field needs the user)",
        )
        expect(
            failures,
            data.get("step", {}).get("id") == "demo_reg_applicant",
            "the step should be the session's own",
        )
        expect(failures, len(db.action_rows) == 5, f"wrote {len(db.action_rows)} action rows")
        expect(failures, len(db.counter_statements) == 1, "expected one counters upsert")
        expect(failures, len(db.session_updates) == 1, "expected one session counter update")
    finally:
        teardown(logs)

    return failures


async def check_rows_come_from_the_plan() -> list[str]:
    """The action type and the field id are the plan's, whatever the request says."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        # The client reports a1 against the wrong field, and a6 which the plan never contained.
        entries = [
            {"action_id": "a1", "field_id": "WRONG", "status": "ok"},
            {"action_id": "a3", "field_id": "f4", "status": "ok"},
            {"action_id": "a6", "field_id": "f9", "status": "failed", "reason": "UNKNOWN_FIELD"},
        ]
        data = data_of(client.post("/results", json=results_body(entries=entries)))

        expect(
            failures,
            data.get("recorded") == 2,
            f"recorded {data.get('recorded')}; an unplanned action id must be skipped",
        )

        written = {(row.action_type, row.field_id, row.status) for row in db.action_rows}

        expect(
            failures,
            ("fill", "f1", "ok") in written,
            "a1's row should carry the plan's own type and field, not the request's",
        )
        expect(
            failures,
            not any(row.field_id == "WRONG" for row in db.action_rows),
            "the client's field id reached a row",
        )
        expect(
            failures,
            not any(row.field_id == "f9" for row in db.action_rows),
            "an action id outside the plan produced a row",
        )
        expect(
            failures,
            ("select", "f4", "ok") in written,
            "a3's row should carry the plan's `select` type",
        )
        # The run totals still count what was reported, including the skipped entry: the user's
        # summary is about their page, not about our bookkeeping.
        expect(failures, data.get("run", {}).get("failed") == 1, "the run totals should count all")
    finally:
        teardown(logs)

    return failures


async def check_reporting_twice_writes_once() -> list[str]:
    """DoD item 2: the same run reported twice changes nothing and answers the same."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        first = data_of(client.post("/results", json=results_body()))
        rows_after_first = len(db.action_rows)
        counters_after_first = len(db.counter_statements)

        second = data_of(client.post("/results", json=results_body()))

        expect(failures, first == second, "the second report answered differently")
        expect(failures, second.get("acknowledged") is True, "a replay should still acknowledge")
        expect(
            failures,
            len(db.action_rows) == rows_after_first,
            "the second report wrote more action rows",
        )
        expect(
            failures,
            len(db.counter_statements) == counters_after_first,
            "the second report incremented the counters again",
        )
        expect(
            failures,
            len(db.session_updates) == 1,
            "the second report bumped the session counters again",
        )
    finally:
        teardown(logs)

    return failures


async def check_concurrent_claim_replays() -> list[str]:
    """A plan claimed between our read and our insert: replay it, write nothing."""
    failures: list[str] = []
    client, db, logs = build_harness()
    db.steal_claim = True

    try:
        data = data_of(client.post("/results", json=results_body()))

        expect(failures, data.get("acknowledged") is True, "the winner's answer should come back")
        expect(failures, db.action_rows == [], "the loser of a claim race wrote action rows")
        expect(failures, db.counter_statements == [], "the loser of a claim race wrote counters")
        expect(failures, db.session_updates == [], "the loser of a claim race bumped the session")
    finally:
        teardown(logs)

    return failures


async def check_unknown_plan_and_session() -> list[str]:
    """Every "we have no record of this" case is a 200 that records nothing."""
    failures: list[str] = []
    cases = [
        ("an unknown session", {"session": None}),
        ("another user's session", {"session": make_session(user_id=OTHER_USER_ID)}),
        ("an unknown or expired plan", {"remember": False}),
        ("a plan from another session", {"plan_session_id": OTHER_SESSION_ID}),
        ("a plan issued to another user", {"plan_user_id": OTHER_USER_ID}),
    ]

    for description, kwargs in cases:
        client, db, logs = build_harness(**kwargs)  # type: ignore[arg-type]

        try:
            response = client.post("/results", json=results_body())
            data = data_of(response)

            expect(
                failures,
                response.status_code == 200,
                f"{description} gave {response.status_code}, not 200",
            )
            expect(
                failures,
                data.get("acknowledged") is False,
                f"{description} was acknowledged",
            )
            expect(failures, data.get("recorded") == 0, f"{description} recorded something")
            expect(
                failures,
                data.get("run", {}) == dict.fromkeys(RunTotals.model_fields, 0),
                f"{description} came back with non-zero counts",
            )
            expect(
                failures,
                data.get("next_hint", {}).get("code") == "REVIEW_CONTINUE",
                f"{description} did not get the default hint",
            )
            expect(failures, db.action_rows == [], f"{description} wrote action rows")
            expect(failures, db.counter_statements == [], f"{description} wrote counters")
        finally:
            teardown(logs)

    return failures


async def check_checkpoint() -> list[str]:
    """DoD item 3: a checkpoint is recorded, normalised for the counters, and outranks the rest."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        body = results_body(
            entries=[
                {"action_id": "a1", "field_id": "f1", "status": "ok"},
                {
                    "action_id": "a2",
                    "field_id": "f2",
                    "status": "failed",
                    "reason": "NOT_ACCEPTED",
                },
                {
                    "action_id": "a3",
                    "field_id": "f4",
                    "status": "cancelled",
                    "reason": "CHECKPOINT",
                },
            ],
            checkpoint={"reason": "One-Time Code", "after_index": 1},
        )
        data = data_of(client.post("/results", json=body))

        expect(
            failures,
            data.get("next_hint", {}).get("code") == "CHECKPOINT_PENDING",
            f"a checkpoint should outrank a failure, got {data.get('next_hint', {}).get('code')}",
        )
        expect(
            failures,
            data.get("next_hint", {}).get("message") == NEXT_HINTS["CHECKPOINT_PENDING"],
            "the checkpoint sentence should be the catalogue's own",
        )
        expect(failures, len(db.events) == 1, f"wrote {len(db.events)} checkpoint events")

        if db.events:
            event = db.events[0]
            expect(failures, event.reason == "One-Time Code", "the event keeps the reported kind")
            expect(failures, event.after_index == 1, "the event keeps its position in the batch")
            expect(failures, event.plan_id == PLAN_ID, "the event names its plan")

        expect(
            failures,
            "checkpoint:one_time_code" in db.counter_metrics,
            f"the counter should use the normalised kind; wrote {sorted(db.counter_metrics)}",
        )
        expect(
            failures,
            "one_time_code" not in {"One-Time Code"} & db.counter_metrics,
            "the reported spelling must not become a counter of its own",
        )
        expect(failures, bool(db.counter_params), "a checkpoint run still increments the counters")
        expect(failures, data.get("run", {}).get("cancelled") == 1, "cancelled should be counted")
    finally:
        teardown(logs)

    return failures


async def check_hint_priority() -> list[str]:
    """Failures outrank finality; a clean final step tells the user to submit it themselves."""
    failures: list[str] = []
    clean = [
        {"action_id": "a1", "field_id": "f1", "status": "ok"},
        {"action_id": "a2", "field_id": "f2", "status": "changed", "reason": "NOT_ACCEPTED"},
    ]
    cases = [
        ("demo_reg_verify", clean, "FINAL_REVIEW"),
        ("demo_reg_applicant", clean, "REVIEW_CONTINUE"),
        (
            "demo_reg_verify",
            [
                *clean,
                {
                    "action_id": "a3",
                    "field_id": "f4",
                    "status": "failed",
                    "reason": "NOT_ACCEPTED",
                },
            ],
            "FIX_FIRST",
        ),
    ]

    for step_id, entries, expected in cases:
        client, _db, logs = build_harness(session=make_session(step_id=step_id))

        try:
            data = data_of(client.post("/results", json=results_body(entries=entries)))
            code = data.get("next_hint", {}).get("code")

            expect(failures, code == expected, f"{step_id} with {len(entries)} results gave {code}")

            if expected == "FINAL_REVIEW":
                expect(
                    failures,
                    "submit it yourself" in data.get("next_hint", {}).get("message", ""),
                    "the final-step sentence must put the submit on the user",
                )
        finally:
            teardown(logs)

    return failures


async def check_unknown_page_still_records() -> list[str]:
    """A page outside the registry records its run, with no step and no invented position."""
    failures: list[str] = []
    client, db, logs = build_harness(
        session=make_session(workflow_id=None, step_id=None),
    )

    try:
        data = data_of(client.post("/results", json=results_body()))

        expect(failures, data.get("acknowledged") is True, "an unknown page should still record")
        expect(failures, data.get("step") is None, "a page with no workflow must get no step")
        expect(failures, len(db.action_rows) == 5, "the rows should still be written")
        expect(
            failures,
            db.counter_scopes == {("-", "-")},
            f"a page with no workflow should count under the sentinel; got {db.counter_scopes}",
        )
    finally:
        teardown(logs)

    return failures


async def check_counter_sql() -> list[str]:
    """The counters upsert on the whole key, and the session's tally is incremented in SQL."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        client.post("/results", json=results_body())
        counters = db.counter_statements[0] if db.counter_statements else ""
        session_update = db.session_updates[0] if db.session_updates else ""

        expect(failures, "ON CONFLICT" in counters.upper(), "the counters write is not an upsert")
        expect(
            failures,
            "workflow_id" in counters and "step_id" in counters and "metric" in counters,
            "the counters conflict target should be the whole key",
        )
        expect(
            failures,
            "day" in counters,
            "the counters must be keyed by day, or yesterday's numbers merge into today's",
        )
        expect(
            failures,
            "count + " in counters or "count +" in counters,
            "the upsert should add its delta rather than replace the count",
        )
        expect(
            failures,
            "results_ok" in session_update and "+" in session_update,
            "the session counters should be incremented in SQL, not read-modify-written",
        )
    finally:
        teardown(logs)

    return failures


async def check_value_tripwire() -> list[str]:
    """A result carrying what was written is refused by name, and the value is never echoed."""
    failures: list[str] = []

    for key in ("value", "text", "label"):
        client, db, logs = build_harness()

        try:
            poisoned = {"action_id": "a1", "field_id": "f1", "status": "ok", key: "hunter2"}
            response = client.post("/results", json=results_body(entries=[poisoned]))
            error = error_of(response)

            expect(failures, response.status_code == 400, f"{key} gave {response.status_code}")
            expect(failures, error.get("code") == "INVALID_REQUEST", f"{key}: wrong error code")
            expect(
                failures,
                key in error.get("message", ""),
                f"{key}: the message should name the key",
            )
            expect(failures, "hunter2" not in response.text, f"{key}: the value was echoed back")
            expect(failures, "hunter2" not in logs.text, f"{key}: the value reached a log line")
            expect(failures, db.action_rows == [], f"{key}: a refused request wrote rows")
        finally:
            teardown(logs)

    return failures


async def check_validation() -> list[str]:
    """Every schema violation in the spec's error table is a 400 that records nothing."""
    failures: list[str] = []
    cases = {
        "an unknown reason code": [
            {"action_id": "a1", "field_id": "f1", "status": "failed", "reason": "WHO_KNOWS"},
        ],
        "a non-ok result with no reason": [
            {"action_id": "a1", "field_id": "f1", "status": "failed"},
        ],
        "a status outside the enum": [
            {"action_id": "a1", "field_id": "f1", "status": "exploded", "reason": "MALFORMED"},
        ],
        "an over-long reason": [
            {"action_id": "a1", "field_id": "f1", "status": "failed", "reason": "X" * 41},
        ],
        "too many results": [
            {"action_id": f"a{index}", "field_id": "f1", "status": "ok"}
            for index in range(MAX_RESULTS + 1)
        ],
    }

    for description, entries in cases.items():
        client, db, logs = build_harness()

        try:
            response = client.post("/results", json=results_body(entries=entries))

            expect(
                failures,
                response.status_code == 400,
                f"{description} gave {response.status_code}, not 400",
            )
            expect(
                failures,
                error_of(response).get("code") == "INVALID_REQUEST",
                f"{description}: wrong error code",
            )
            expect(failures, db.action_rows == [], f"{description} wrote rows anyway")
        finally:
            teardown(logs)

    # And the batch-level fields.
    for description, extra in {
        "a negative elapsed_ms": {"elapsed_ms": -1},
        "an absurd elapsed_ms": {"elapsed_ms": 10**9},
        "an unknown abort reason": {"aborted": "because"},
        "a negative checkpoint index": {"checkpoint": {"reason": "otp", "after_index": -1}},
        "an unknown key": {"filled": 9},
    }.items():
        client, _db, logs = build_harness()

        try:
            response = client.post("/results", json=results_body(**extra))

            expect(
                failures,
                response.status_code == 400,
                f"{description} gave {response.status_code}, not 400",
            )
        finally:
            teardown(logs)

    return failures


async def check_abort_is_recorded() -> list[str]:
    """An aborted batch is recorded with its reason, as its own counter."""
    failures: list[str] = []
    client, db, logs = build_harness()

    try:
        body = results_body(
            entries=[
                {
                    "action_id": "a1",
                    "field_id": "f1",
                    "status": "cancelled",
                    "reason": "STALE_PAGE",
                },
            ],
            aborted="stale_page",
        )
        data = data_of(client.post("/results", json=body))

        expect(failures, data.get("acknowledged") is True, "an aborted run should still record")
        expect(failures, data.get("run", {}).get("cancelled") == 1, "the cancelled action counts")
        expect(
            failures,
            "abort:stale_page" in db.counter_metrics,
            f"an abort should get its own counter metric; wrote {sorted(db.counter_metrics)}",
        )
    finally:
        teardown(logs)

    return failures


async def check_rate_limit() -> list[str]:
    """Sixty a minute per token, then a 429 carrying the wait."""
    failures: list[str] = []
    client, _db, logs = build_harness()

    try:
        statuses = [
            client.post("/results", json=results_body()).status_code
            for _ in range(RATE_LIMIT + 1)
        ]

        expect(
            failures,
            all(code == 200 for code in statuses[:RATE_LIMIT]),
            "a request inside the limit was refused",
        )
        expect(failures, statuses[-1] == 429, f"the request past the limit gave {statuses[-1]}")
        expect(
            failures,
            error_of(client.post("/results", json=results_body())).get("retry_after", 0) > 0,
            "no retry_after on the 429",
        )
    finally:
        teardown(logs)

    return failures


async def check_statuses_are_in_sync() -> list[str]:
    """
    The status vocabulary exists three times and all three have to agree.

    The request's `Literal`, the tuple `/results` validates against, and the tuple the database
    CHECK mirrors. A widened enum with a forgotten migration is a 500 on the first row written.
    """
    failures: list[str] = []

    expect(failures, statuses_are_in_sync(), "ResultStatus and RESULT_STATUSES disagree")
    expect(
        failures,
        set(ACTION_STATUSES) == {"ok", "changed", "failed", "rejected", "cancelled"},
        f"ACTION_STATUSES is {ACTION_STATUSES}; the column CHECK would refuse a valid report",
    )

    return failures


async def check_logs_are_content_free() -> list[str]:
    """Not a label, not a value, not a source sentence — and the request line is emitted."""
    failures: list[str] = []
    client, _db, logs = build_harness()

    try:
        client.post(
            "/results",
            json=results_body(checkpoint={"reason": "one_time_code", "after_index": 2}),
        )
        text = logs.text

        forbidden = {
            "a field label": "SECRETLABEL77777",
            "a profile value": "Adaeze Okonkwo",
            "an email": "adaeze@example.com",
            "a chat value": "Ikeja",
            "a provenance sentence": "You told me just now",
        }

        for description, planted in forbidden.items():
            expect(failures, planted not in text, f"{description} appeared in a log line")

        expect(failures, "results user=" in text, "the request line was not emitted")
        expect(failures, "acknowledged=True" in text, "the request line omits the outcome")
        expect(failures, "hint=" in text, "the request line omits the hint")
        expect(failures, "elapsed_ms=" in text, "the request line omits the run's own duration")
    finally:
        teardown(logs)

    return failures


async def check_health_block() -> list[str]:
    """`/health` publishes the counters, and degrades to null rather than failing."""
    failures: list[str] = []

    class Rows:
        def all(self) -> list[Any]:
            @dataclass
            class Row:
                day: Any
                workflow_id: str
                step_id: str
                metric: str
                count: int

            return [
                Row(
                    datetime(2026, 9, 27, tzinfo=UTC).date(),
                    "demo_reg",
                    "demo_reg_applicant",
                    "ok",
                    7,
                ),
            ]

    class OkDb:
        async def execute(self, _statement: Any) -> Rows:
            return Rows()

        async def __aenter__(self) -> "OkDb":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class BrokenDb:
        async def execute(self, _statement: Any) -> Rows:
            message = "connection refused"
            raise RuntimeError(message)

        async def __aenter__(self) -> "BrokenDb":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    from src.cache import invalidate

    invalidate()
    block = await results_health.results_health(lambda: OkDb())  # type: ignore[arg-type]

    expect(failures, block is not None and len(block) == 1, "the counters should be reported")

    if block:
        expect(failures, block[0]["metric"] == "ok", "the metric should be reported")
        expect(failures, block[0]["count"] == 7, "the count should be reported")  # noqa: PLR2004
        expect(failures, block[0]["day"] == "2026-09-27", "the day should be an ISO date")

    invalidate()
    broken = await results_health.results_health(lambda: BrokenDb())  # type: ignore[arg-type]

    expect(failures, broken is None, "a database failure should degrade to null, not an empty list")

    return failures


CHECKS: list[tuple[str, Any]] = [
    ("a run is recorded once, with its hint", check_records_a_run),
    ("rows come from the plan, not the request", check_rows_come_from_the_plan),
    ("reporting twice writes once", check_reporting_twice_writes_once),
    ("a concurrent claim replays, never doubles", check_concurrent_claim_replays),
    ("unknown session or plan is a 200", check_unknown_plan_and_session),
    ("a checkpoint is recorded and outranks", check_checkpoint),
    ("the hint table's priority order holds", check_hint_priority),
    ("an unknown page still records", check_unknown_page_still_records),
    ("the counter SQL upserts and increments", check_counter_sql),
    ("a result carrying content is refused", check_value_tripwire),
    ("every schema violation is a 400", check_validation),
    ("an aborted batch is recorded", check_abort_is_recorded),
    ("the rate limit holds", check_rate_limit),
    ("the status vocabulary is in sync", check_statuses_are_in_sync),
    ("no content in any log line", check_logs_are_content_free),
    ("the health block reports and degrades", check_health_block),
]


async def main() -> int:
    configure_logging()
    logging.getLogger().setLevel(logging.INFO)

    failures: list[str] = []

    print(f"checks ({len(CHECKS)}):")

    for name, func in CHECKS:
        try:
            case_failures = await func()
        except Exception as exc:  # noqa: BLE001  # a check that raised is a check that failed
            case_failures = [f"raised {type(exc).__name__}: {exc}"]

        print(f"  {'FAIL' if case_failures else 'ok  '}  {name}")
        failures += [f"{name}: {line}" for line in case_failures]

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for line in failures:
            print(f"  - {line}")

        return 1

    print("\nAll results checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
