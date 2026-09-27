"""
Check `POST /plan` without a model, a network, a database, or pytest.

    python -m scripts.check_plan

Everything runs over the real app, through the real middleware and the real exception handlers —
so what these checks read is the body a caller would actually receive, wrapped shape and all — with
three things replaced: the database session (a stub that answers the four calls this endpoint
makes), the model (`src/ai/fake.py`, replaying a scripted reply), and retrieval (a stub returning a
fixed chunk).

What it covers, against P5's own test list:

1.  a valid request returns actions whose values come from the profile, with provenance
2.  a snapshot that no longer matches the session is `PAGE_CHANGED`, and no model call is made
3.  the same request twice inside the window is served from the store — one model call in total
4.  a different message is not cached
5.  `extracted_data` lands in the session's `chat_values` and never in the profile
6.  history is trimmed to ten turns
7.  approved actions are retrievable by `plan_id`, and not after their expiry
8.  a sensitive field is rejected with a readable sentence and never appears in `actions`
9.  a required field with no data appears in `missing`
10. a failed turn is the right status with a readable sentence, and nothing is persisted
11. a second turn for one session is `SESSION_BUSY`
12. another user's session is `SESSION_NOT_FOUND`, not `403`
13. a field object carrying `value` is a 400 naming the key
14. no message, reply, label, value or retrieved content in any log line

What it cannot cover, and what has to be done by hand: the migration, the real session row, the
6-second median, and whether a real model's output maps onto the preview sensibly.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from src.agent.breaker import reset_breaker
from src.agent.locks import reset_locks
from src.agent.quota import reset_quota
from src.ai.client import ModelPermanentError
from src.ai.fake import FakeModelClient, answer, searches
from src.auth.schemas import AuthedUser
from src.database.session import get_session
from src.documents.schemas import RetrievalResult, RetrievedChunk
from src.logging import RedactingFilter, configure_logging
from src.main import app
from src.plan import service as plan_service
from src.plan import store
from src.plan.constants import (
    IDEMPOTENCY_WINDOW_SECONDS,
    PLAN_TTL_SECONDS,
    RATE_LIMIT,
)
from src.profiles.models import Profile
from src.rate_limit import reset_rate_limits
from src.sessions.constants import MAX_HISTORY_TURNS
from src.sessions.models import Session
from src.tokens.dependencies import require_token
from src.turn_errors import FAILURE_STATUS
from src.workflows import service as workflows_service
from src.workflows.schemas import Step

# --- The page, the user, and the corpus every case runs against -------------------------------
#
# Labels are the seeded `demo_reg` applicant step's, so a mismatch here is a real mismatch.

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_USER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
TOKEN_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")

URL = "https://portal.example.gov.ng/register/applicant"

FIELDS: list[dict[str, Any]] = [
    {"field_id": "f1", "label": "Full Name", "type": "text", "required": True},
    {"field_id": "f2", "label": "Email Address", "type": "email", "required": True},
    {"field_id": "f3", "label": "Phone Number", "type": "tel", "required": True},
    {
        "field_id": "f4",
        "label": "State",
        "type": "select",
        "required": True,
        "options": ["Abia", "Kano", "Lagos", "Rivers"],
    },
    {"field_id": "f5", "label": "Local Government Area", "type": "text", "required": True},
    {
        "field_id": "f6",
        "label": "Password",
        "type": "password",
        "required": True,
        "sensitive": True,
    },
]
BUTTONS: list[dict[str, Any]] = [
    {"field_id": "b1", "text": "Continue"},
    {"field_id": "b2", "text": "Submit Application", "sensitive": True},
]

PROFILE_VALUES = {
    "full_name": "Adaeze Okonkwo",
    "email": "adaeze@example.com",
    "phone": "08000000000",
    "address": "12 Marina Road, Lagos Island",
    "state": "Lagos",
    "lga": None,
}

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

CHUNK = RetrievedChunk(
    chunk_id=7,
    title="Demo Agency FAQ — registration",
    content="The registration fee is 5,000 naira.",
    source_url="https://example.gov.ng/faq",
    agency="Demo Agency",
    service="Demo Service",
    distance=0.2,
    ingested_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
)

# Planted strings. None of these may appear in any log line.
PLANTED_MESSAGE = "Fill this in for me, my SECRETQUESTION12345"
PLANTED_REPLY = "I filled what I had, SECRETREPLY98765."

EMPTY_EXTRACTED: dict[str, Any] = dict.fromkeys(PROFILE_VALUES)


def plan_answer(
    *,
    reply: str = PLANTED_REPLY,
    actions: list[dict[str, Any]] | None = None,
    extracted: dict[str, Any] | None = None,
    citations: list[dict[str, Any]] | None = None,
    missing: list[dict[str, Any]] | None = None,
) -> Any:  # noqa: ANN401  # a ModelReply, from the fake's own helper
    """One scripted model answer in `PlanResponse`'s shape."""
    return answer(
        {
            "reply": reply,
            "actions": actions or [],
            "extracted_data": {**EMPTY_EXTRACTED, **(extracted or {})},
            "citations": citations or [],
            "missing": missing or [],
        },
    )


FILL_ACTIONS: list[dict[str, Any]] = [
    {"type": "fill", "field_id": "f2", "value_ref": {"source": "profile", "key": "email"}},
    {"type": "fill", "field_id": "f6", "value_ref": {"source": "profile", "key": "email"}},
]


# --- The stub database ------------------------------------------------------------------------


@dataclass
class StubResult:
    """What `execute` returns: one row or none."""

    row: Any = None

    def scalar_one_or_none(self) -> Any:  # noqa: ANN401  # a Session, or None
        return self.row


@dataclass
class StubSession:
    """
    The four calls `/plan` makes on a database session, and a record of what was written.

    Faithful where it matters: `execute` returns the session row only when the statement's
    user id matches, which is what makes the "another user's session" case a real check of the
    WHERE clause rather than of this stub's opinion.
    """

    session: Session | None
    profile: Profile
    flushes: int = 0
    added: list[Any] = field(default_factory=list)
    committed: int = 0
    rolled_back: int = 0

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> StubResult:
        wanted = _user_id_in(statement)

        if self.session is None or (wanted is not None and wanted != self.session.user_id):
            return StubResult(None)

        return StubResult(self.session)

    async def get(self, _model: Any, _key: Any) -> Profile:
        return self.profile

    async def flush(self) -> None:
        self.flushes += 1

    def add_all(self, rows: Sequence[Any]) -> None:
        self.added.extend(rows)

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back += 1


def _user_id_in(statement: Any) -> uuid.UUID | None:
    """The user id a compiled SELECT is filtering on, so the stub can honour the scope."""
    try:
        params = statement.compile().params
    except Exception:  # noqa: BLE001  # a statement the stub does not need to understand
        return None

    for value in params.values():
        if isinstance(value, uuid.UUID) and value in (USER_ID, OTHER_USER_ID):
            return value

    return None


def make_session(  # noqa: PLR0913  # one keyword per thing a case may vary
    *,
    page_hash: str,
    user_id: uuid.UUID = USER_ID,
    history: list[dict[str, Any]] | None = None,
    chat_values: dict[str, str] | None = None,
    workflow_id: str | None = "demo_reg",
    step_id: str | None = "demo_reg_applicant",
) -> Session:
    """An unsaved session row, as the database would have handed it back."""
    session = Session(
        user_id=user_id,
        tab_id=1,
        workflow_id=workflow_id,
        step_id=step_id,
        page_hash=page_hash,
        cached_rule_ids=[],
        history=history if history is not None else [],
        chat_values=chat_values if chat_values is not None else {},
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    session.id = SESSION_ID

    return session


def make_profile(values: dict[str, Any] | None = None) -> Profile:
    """A profile row carrying the demo account's fictional details."""
    profile = Profile(user_id=USER_ID, **(values if values is not None else PROFILE_VALUES))
    profile.version = 3

    return profile


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


@dataclass
class Harness:
    """One request's worth of stubs: the database, the model, retrieval, and the log."""

    db: StubSession
    model: FakeModelClient
    logs: LogCapture
    retrieval_calls: int = 0


def page_hash_for(fields: list[dict[str, Any]] | None = None) -> str:
    """The hash the server will compute for this snapshot."""
    from src.context.schemas import PageField
    from src.context.utils import compute_page_hash

    parsed = [PageField.model_validate(item) for item in (fields or FIELDS)]

    return compute_page_hash(URL, parsed)


def body(
    *,
    message: str = PLANTED_MESSAGE,
    fields: list[dict[str, Any]] | None = None,
    session_id: uuid.UUID = SESSION_ID,
    hash_fields: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    A well-formed request body, with room to break one thing at a time.

    `hash_fields` exists because the client's `page_hash` is computed here through the real
    `PageField` — so a case that deliberately poisons a field cannot hash its own snapshot
    without tripping the same tripwire it is testing. It hashes the clean fields instead.
    """
    return {
        "session_id": str(session_id),
        "url": URL,
        "page_hash": page_hash_for(hash_fields if hash_fields is not None else fields),
        "message": message,
        "fields": fields if fields is not None else FIELDS,
        "buttons": BUTTONS,
        **extra,
    }


def build_harness(
    script: Sequence[Any],
    *,
    session: Session | None,
    profile: Profile | None = None,
    retrieval: RetrievalResult | None = None,
) -> tuple[TestClient, Harness]:
    """
    The app with its three edges replaced, and a client to call it with.

    Global state — the rate limiter, the turn locks, the quota, the breaker and both plan stores
    — is reset here, so a case cannot pass or fail because of the case before it.
    """
    reset_rate_limits()
    reset_locks()
    reset_quota()
    reset_breaker()
    store.reset_plans()

    db = StubSession(session=session, profile=profile or make_profile())
    model = FakeModelClient(script=list(script))
    logs = LogCapture()
    harness = Harness(db=db, model=model, logs=logs)

    async def stub_retrieve(*_args: Any, **_kwargs: Any) -> RetrievalResult:
        harness.retrieval_calls += 1

        return retrieval or RetrievalResult(chunks=[CHUNK], available=True)

    async def stub_steps(*_args: Any, **_kwargs: Any) -> list[Step]:
        return STEPS

    plan_service.shared_client = lambda: model  # type: ignore[assignment]
    plan_service.search_government_information = stub_retrieve  # type: ignore[assignment]
    workflows_service.get_steps = stub_steps  # type: ignore[assignment]

    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_token] = lambda: AuthedUser(
        id=USER_ID,
        email="demo@example.com",
        token_id=TOKEN_ID,
    )

    root = logging.getLogger()
    root.addHandler(logs)

    client = TestClient(app, raise_server_exceptions=False)

    return client, harness


def teardown(logs: LogCapture) -> None:
    logging.getLogger().removeHandler(logs)
    app.dependency_overrides.clear()


def expect(failures: list[str], condition: bool, description: str) -> None:  # noqa: FBT001
    if not condition:
        failures.append(description)


# --- Cases -----------------------------------------------------------------------------------


async def check_happy_path() -> list[str]:
    """A fill turn: values from the profile, provenance on every row, the step named."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        [plan_answer(actions=[FILL_ACTIONS[0]], extracted={"lga": "Ikeja"})],
        session=session,
    )

    try:
        response = client.post("/plan", json=body())
        payload = response.json()
        data = payload.get("data", {})

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(failures, payload.get("success") is True, "response was not a success envelope")
        expect(failures, bool(data.get("plan_id")), "no plan_id was returned")
        expect(failures, data.get("cached") is False, "a first request was marked cached")

        actions = data.get("actions", [])
        expect(failures, len(actions) == 1, f"expected one action, got {len(actions)}")

        if actions:
            row = actions[0]
            expect(failures, row["action_id"] == "a1", "the first action should be a1")
            expect(
                failures,
                row["value"] == PROFILE_VALUES["email"],
                "the value did not come from the profile",
            )
            expect(failures, row["source"] == "Your profile", "provenance was not worded")
            expect(
                failures,
                row["source_ref"] == "profile.email",
                "the machine-readable source is missing",
            )
            expect(failures, row["label"] == "Email Address", "the row carries no label")

        expect(
            failures,
            data.get("step", {}).get("id") == "demo_reg_applicant",
            "the step was not reported",
        )
        expect(failures, data.get("step", {}).get("total") == 2, "step total should be 2")
        expect(failures, harness.model.call_count == 1, "expected exactly one model call")
        expect(failures, session.chat_values.get("lga") == "Ikeja", "chat value was not stored")
        expect(
            failures,
            len(session.history) == 2,
            f"history should hold two turns, holds {len(session.history)}",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_extracted_data_stays_out_of_the_profile() -> list[str]:
    """A value given in chat reaches the session and nothing else."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    profile = make_profile()
    client, harness = build_harness(
        [plan_answer(extracted={"lga": "Ikeja", "phone": "08111111111"})],
        session=session,
        profile=profile,
    )

    try:
        response = client.post("/plan", json=body())

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(failures, session.chat_values.get("lga") == "Ikeja", "chat value was not stored")
        expect(
            failures,
            profile.lga is None,
            "the profile was written to — a chat message is not a consent screen",
        )
        expect(
            failures,
            profile.phone == PROFILE_VALUES["phone"],
            "an extracted value overwrote a stored profile value",
        )
        expect(
            failures,
            not any(isinstance(row, Profile) for row in harness.db.added),
            "a profile row was added during a plan turn",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_history_is_trimmed() -> list[str]:
    """Ten turns in, two added, ten kept — the oldest go."""
    failures: list[str] = []
    history = [{"role": "user", "content": f"turn {index}"} for index in range(MAX_HISTORY_TURNS)]
    session = make_session(page_hash=page_hash_for(), history=history)
    client, harness = build_harness([plan_answer()], session=session)

    try:
        client.post("/plan", json=body())

        expect(
            failures,
            len(session.history) == MAX_HISTORY_TURNS,
            f"history holds {len(session.history)}, not {MAX_HISTORY_TURNS}",
        )
        expect(
            failures,
            session.history[-1]["content"] == PLANTED_REPLY,
            "the reply was not the last turn kept",
        )
        expect(
            failures,
            all(turn["content"] != "turn 0" for turn in session.history),
            "the oldest turn survived the trim",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_page_changed() -> list[str]:
    """A snapshot that hashes differently is refused, and costs no model call."""
    failures: list[str] = []
    session = make_session(page_hash="a-hash-from-a-different-page")
    client, harness = build_harness([plan_answer()], session=session)

    try:
        response = client.post("/plan", json=body())
        payload = response.json()
        error = payload.get("error", {})

        expect(failures, response.status_code == 409, f"status was {response.status_code}")
        expect(failures, error.get("code") == "PAGE_CHANGED", f"code was {error.get('code')}")
        expect(failures, "page changed" in error.get("message", "").lower(), "no readable sentence")
        expect(failures, harness.model.call_count == 0, "a model call was made anyway")
        expect(failures, harness.db.flushes == 0, "something was written on a refused request")
        expect(failures, session.history == [], "history was written on a refused request")
    finally:
        teardown(harness.logs)

    return failures


async def check_idempotent_retry() -> list[str]:
    """The same question twice: one model call, the same plan id, `cached` the second time."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer(actions=[FILL_ACTIONS[0]])], session=session)

    try:
        first = client.post("/plan", json=body()).json().get("data", {})
        second = client.post("/plan", json=body()).json().get("data", {})

        expect(failures, harness.model.call_count == 1, "a second model call was made")
        expect(failures, first.get("cached") is False, "the first answer was marked cached")
        expect(failures, second.get("cached") is True, "the retry was not marked cached")
        expect(
            failures,
            first.get("plan_id") == second.get("plan_id"),
            "the retry invented a new plan id, so an approval would not match",
        )
        expect(
            failures,
            first.get("actions") == second.get("actions"),
            "the retry returned different actions",
        )
        expect(
            failures,
            len(session.history) == 2,
            "the retry wrote a second pair of history turns",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_different_message_is_not_cached() -> list[str]:
    """A different question is a different plan."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        [plan_answer(), plan_answer(reply="A second answer.")],
        session=session,
    )

    try:
        first = client.post("/plan", json=body()).json().get("data", {})
        second = client.post("/plan", json=body(message="What does LGA mean?"))
        second_data = second.json().get("data", {})

        expect(failures, harness.model.call_count == 2, "the second question reused the first plan")
        expect(failures, second_data.get("cached") is False, "a new question was served as cached")
        expect(
            failures,
            first.get("plan_id") != second_data.get("plan_id"),
            "two different plans share one id",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_whitespace_retry_is_cached() -> list[str]:
    """A retry differing only in whitespace is the same question."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer()], session=session)

    try:
        client.post("/plan", json=body())
        again = client.post("/plan", json=body(message=f"  {PLANTED_MESSAGE}  "))

        expect(failures, harness.model.call_count == 1, "a whitespace-only difference cost a call")
        expect(failures, again.json().get("data", {}).get("cached") is True, "not served as cached")
    finally:
        teardown(harness.logs)

    return failures


async def check_sensitive_field_is_rejected() -> list[str]:
    """The password field is refused with a sentence, and never appears as an action."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer(actions=FILL_ACTIONS)], session=session)

    try:
        data = client.post("/plan", json=body()).json().get("data", {})
        actions = data.get("actions", [])
        rejected = data.get("rejected", [])

        expect(
            failures,
            all(row["field_id"] != "f6" for row in actions),
            "the sensitive field was approved",
        )
        expect(failures, len(rejected) == 1, f"expected one refusal, got {len(rejected)}")

        if rejected:
            expect(failures, rejected[0]["field_id"] == "f6", "the wrong field was refused")
            expect(failures, rejected[0]["label"] == "Password", "the refusal carries no label")
            expect(
                failures,
                "password" in rejected[0]["reason"].lower(),
                "the refusal is not a sentence a person can read",
            )
            expect(
                failures,
                "code" not in rejected[0],
                "a rejection code leaked into the panel's row",
            )

        expect(
            failures,
            all(
                item["field_id"] != "f6"
                for item in data.get("missing", [])
            ),
            "the panel was asked to ask for a password",
        )

        logged = [row for row in harness.db.added if getattr(row, "status", "") == "rejected"]
        expect(failures, len(logged) == 1, "the refusal was not written to action_log")

        if logged:
            expect(
                failures,
                logged[0].reason == "SENSITIVE_FIELD",
                "action_log should carry the code, not prose",
            )
            expect(
                failures,
                not hasattr(logged[0], "value"),
                "action_log grew a value column",
            )
    finally:
        teardown(harness.logs)

    return failures


async def check_missing_is_asked_for() -> list[str]:
    """A required field with nothing behind it becomes a question."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer(actions=[FILL_ACTIONS[0]])], session=session)

    try:
        data = client.post("/plan", json=body()).json().get("data", {})
        missing = data.get("missing", [])
        asked = {item["field_id"] for item in missing}

        expect(failures, "f5" in asked, "the empty LGA field was not asked about")
        expect(failures, "f2" not in asked, "a field we just filled was asked about")
        expect(failures, "f1" not in asked, "a field already in the profile was asked about")

        for item in missing:
            expect(
                failures,
                item["question"].endswith("?"),
                "a missing entry is not phrased as a question",
            )
    finally:
        teardown(harness.logs)

    return failures


async def check_citations_carry_title_and_date() -> list[str]:
    """A grounded answer's citation shows the source's own title and ingest date."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        [
            searches("what is the registration fee"),
            plan_answer(
                reply="The registration fee is 5,000 naira.",
                citations=[{"chunk_id": 7, "source_url": CHUNK.source_url}],
            ),
        ],
        session=session,
    )

    try:
        data = client.post("/plan", json=body(message="What is the fee?")).json().get("data", {})
        citations = data.get("citations", [])

        expect(failures, harness.retrieval_calls == 1, "retrieval was not called")
        expect(failures, data.get("grounding") == "grounded", f"grounding {data.get('grounding')}")
        expect(failures, len(citations) == 1, f"expected one citation, got {len(citations)}")

        if citations:
            expect(failures, citations[0]["title"] == CHUNK.title, "citation title is wrong")
            expect(failures, citations[0]["url"] == CHUNK.source_url, "citation url is wrong")
            expect(
                failures,
                citations[0]["retrieved_at"] == "2026-09-20",
                "the citation shows the wrong date — it must be the corpus's, not today's",
            )
    finally:
        teardown(harness.logs)

    return failures


async def check_pending_plan_lifetime() -> list[str]:
    """Approved actions are retrievable by plan id, and not after ten minutes."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer(actions=[FILL_ACTIONS[0]])], session=session)

    try:
        data = client.post("/plan", json=body()).json().get("data", {})
        plan_id = data.get("plan_id", "")

        held = store.get_pending_plan(plan_id, USER_ID)
        expect(failures, held is not None, "the plan was not held for approval")

        if held is not None:
            expect(failures, len(held.actions) == 1, "the held plan has the wrong actions")
            expect(
                failures,
                held.actions[0].value == PROFILE_VALUES["email"],
                "the held action lost its value",
            )

        expect(
            failures,
            store.get_pending_plan(plan_id, OTHER_USER_ID) is None,
            "another user could read the held plan",
        )
        expect(
            failures,
            store.get_pending_plan("not-a-plan-id", USER_ID) is None,
            "an unknown plan id returned something",
        )

        # Ten minutes on. The store reads the monotonic clock, so the clock is what moves.
        real_monotonic = time.monotonic
        store.time.monotonic = lambda: real_monotonic() + PLAN_TTL_SECONDS + 1  # type: ignore[attr-defined]

        try:
            expect(
                failures,
                store.get_pending_plan(plan_id, USER_ID) is None,
                "an expired plan is still applicable",
            )
        finally:
            store.time.monotonic = real_monotonic  # type: ignore[attr-defined]
    finally:
        teardown(harness.logs)

    return failures


async def check_idempotency_window_expires() -> list[str]:
    """Past the window, the same question is asked again."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer(), plan_answer()], session=session)

    try:
        client.post("/plan", json=body())

        real_monotonic = time.monotonic
        store.time.monotonic = lambda: real_monotonic() + IDEMPOTENCY_WINDOW_SECONDS + 1  # type: ignore[attr-defined]

        try:
            again = client.post("/plan", json=body())
        finally:
            store.time.monotonic = real_monotonic  # type: ignore[attr-defined]

        expect(failures, harness.model.call_count == 2, "the stale cache entry was served")
        expect(failures, again.json().get("data", {}).get("cached") is False, "marked cached")
    finally:
        teardown(harness.logs)

    return failures


async def check_session_not_found() -> list[str]:
    """A missing session and another user's are the same 404."""
    failures: list[str] = []

    for description, session in (
        ("no session row", None),
        ("another user's session", make_session(page_hash=page_hash_for(), user_id=OTHER_USER_ID)),
    ):
        client, harness = build_harness([plan_answer()], session=session)

        try:
            response = client.post("/plan", json=body())
            error = response.json().get("error", {})

            expect(
                failures,
                response.status_code == 404,
                f"{description}: status was {response.status_code}, not 404",
            )
            expect(
                failures,
                error.get("code") == "SESSION_NOT_FOUND",
                f"{description}: code was {error.get('code')}",
            )
            expect(
                failures,
                bool(error.get("message")),
                f"{description}: no sentence for the user",
            )
            expect(
                failures,
                harness.model.call_count == 0,
                f"{description}: a model call was made",
            )
        finally:
            teardown(harness.logs)

    return failures


async def check_failure_mapping() -> list[str]:
    """Every runner failure has a status and a sentence, and writes nothing."""
    failures: list[str] = []

    expect(
        failures,
        set(FAILURE_STATUS) == {
            "MODEL_UNAVAILABLE",
            "MODEL_TIMEOUT",
            "BUDGET_EXCEEDED",
            "SCHEMA_FAILED",
            "SESSION_BUSY",
            "QUOTA_EXCEEDED",
        },
        "the failure table no longer matches the runner's codes",
    )

    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([ModelPermanentError("no key")], session=session)

    try:
        response = client.post("/plan", json=body())
        error = response.json().get("error", {})

        expect(failures, response.status_code == 503, f"status was {response.status_code}")
        expect(
            failures,
            error.get("code") == "MODEL_UNAVAILABLE",
            f"code was {error.get('code')}",
        )
        expect(
            failures,
            "try again" in error.get("message", "").lower(),
            "the failure sentence does not tell the user what to do",
        )
        expect(failures, session.history == [], "history was written for a failed turn")
        expect(failures, harness.db.flushes == 0, "something was flushed for a failed turn")
        expect(failures, harness.db.added == [], "action_log was written for a failed turn")
        expect(failures, store.sizes().pending == 0, "a failed turn left a pending plan")
        expect(failures, store.sizes().answered == 0, "a failed turn was cached")
    finally:
        teardown(harness.logs)

    return failures


async def check_schema_failure_is_plan_failed() -> list[str]:
    """An answer that never satisfies the schema is a 502, after one repair."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        # Three, not two. While tools are still on offer a reply that does not parse is "not
        # the answer yet": the runner withdraws tools and asks again (call 2), and only that
        # answer failing earns the one repair (call 3). A script of two would exhaust the fake
        # and raise, which is `ScriptExhaustedError` doing its job rather than a finding.
        [answer("not json at all"), answer("still not json"), answer("nor is this")],
        session=session,
    )

    try:
        response = client.post("/plan", json=body())
        error = response.json().get("error", {})

        expect(failures, response.status_code == 502, f"status was {response.status_code}")
        expect(failures, error.get("code") == "PLAN_FAILED", f"code was {error.get('code')}")
        expect(failures, bool(error.get("message")), "no sentence for the user")
        expect(failures, session.history == [], "history was written for a failed turn")
        expect(failures, harness.model.call_count == 3, "the repair pass did not run")
    finally:
        teardown(harness.logs)

    return failures


async def check_session_busy() -> list[str]:
    """A second turn while one is running is refused, not queued."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer()], session=session)

    try:
        from src.agent import locks

        async with locks.turn_lock(str(SESSION_ID)):
            response = client.post("/plan", json=body())

        error = response.json().get("error", {})

        expect(failures, response.status_code == 409, f"status was {response.status_code}")
        expect(failures, error.get("code") == "SESSION_BUSY", f"code was {error.get('code')}")
        expect(
            failures,
            "moment" in error.get("message", "").lower(),
            "the sentence does not ask the user to wait",
        )
        expect(failures, session.history == [], "a refused turn wrote history")
    finally:
        teardown(harness.logs)

    return failures


async def check_value_tripwire() -> list[str]:
    """A field object carrying what the user typed is a 400 that names the key."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer()], session=session)

    try:
        fields = [{**FIELDS[0], "value": "hunter2"}, *FIELDS[1:]]
        response = client.post("/plan", json=body(fields=fields, hash_fields=FIELDS))
        error = response.json().get("error", {})
        message = error.get("message", "")

        expect(failures, response.status_code == 400, f"status was {response.status_code}")
        expect(failures, error.get("code") == "INVALID_REQUEST", f"code was {error.get('code')}")
        expect(failures, "value" in message, "the message does not name the offending key")
        expect(failures, "hunter2" not in message, "the rejected value was echoed back")
        expect(failures, "hunter2" not in harness.logs.text, "the rejected value was logged")
        expect(failures, harness.model.call_count == 0, "a model call was made")
    finally:
        teardown(harness.logs)

    return failures


async def check_over_long_message() -> list[str]:
    """A message past the cap is refused rather than truncated."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness([plan_answer()], session=session)

    try:
        response = client.post("/plan", json=body(message="x" * 2001))

        expect(failures, response.status_code == 400, f"status was {response.status_code}")
        expect(failures, harness.model.call_count == 0, "an over-long message reached the model")
        expect(failures, "xxxxxxxxxx" not in harness.logs.text, "the message body was logged")
    finally:
        teardown(harness.logs)

    return failures


async def check_rate_limit() -> list[str]:
    """Past twenty plans a minute, the token is asked to wait."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        [plan_answer(reply=f"answer {index}") for index in range(RATE_LIMIT + 1)],
        session=session,
    )

    try:
        statuses = [
            client.post("/plan", json=body(message=f"question {index}")).status_code
            for index in range(RATE_LIMIT + 1)
        ]

        expect(
            failures,
            statuses[:RATE_LIMIT] == [200] * RATE_LIMIT,
            f"a request inside the limit was refused: {statuses}",
        )
        expect(failures, statuses[-1] == 429, f"the limit was not enforced: {statuses[-1]}")

        last = client.post("/plan", json=body(message="one more"))
        expect(
            failures,
            "retry_after" in last.json().get("error", {}),
            "a 429 does not say how long to wait",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_unsupported_page_still_plans() -> list[str]:
    """A page outside the registry gets a plan and no invented step."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for(), workflow_id=None, step_id=None)
    client, harness = build_harness([plan_answer(actions=[FILL_ACTIONS[0]])], session=session)

    try:
        response = client.post("/plan", json=body())
        data = response.json().get("data", {})

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(failures, data.get("step") is None, "a step was invented for an unknown page")
        expect(failures, len(data.get("actions", [])) == 1, "no plan was produced")
    finally:
        teardown(harness.logs)

    return failures


async def check_logs_are_content_free() -> list[str]:
    """No message, reply, label, value or retrieved content in any log line."""
    failures: list[str] = []
    session = make_session(page_hash=page_hash_for())
    client, harness = build_harness(
        [
            searches("what is the registration fee"),
            plan_answer(
                reply=PLANTED_REPLY,
                actions=FILL_ACTIONS,
                extracted={"lga": "Ikeja"},
                citations=[{"chunk_id": 7, "source_url": CHUNK.source_url}],
            ),
        ],
        session=session,
    )

    try:
        client.post(
            "/plan",
            json=body(headings=["Business Registration — SECRETHEADING55555"]),
        )
        text = harness.logs.text

        forbidden = {
            "the message": "SECRETQUESTION12345",
            "the reply": "SECRETREPLY98765",
            "a heading": "SECRETHEADING55555",
            "a profile value": PROFILE_VALUES["email"],
            "a name": PROFILE_VALUES["full_name"],
            "a chat value": "Ikeja",
            "a field label": "Local Government Area",
            "retrieved content": CHUNK.content,
        }

        for description, planted in forbidden.items():
            expect(failures, planted not in text, f"{description} appeared in a log line")

        expect(failures, "plan user=" in text, "the request line was not emitted")
        expect(failures, "message_chars=" in text, "the request line omits the message length")
        expect(failures, "turn kind=plan" in text, "the runner's turn line was not emitted")
    finally:
        teardown(harness.logs)

    return failures


CHECKS: list[tuple[str, Any]] = [
    ("happy path: values, provenance, step", check_happy_path),
    ("chat values never reach the profile", check_extracted_data_stays_out_of_the_profile),
    ("history trimmed to ten turns", check_history_is_trimmed),
    ("a changed page is refused", check_page_changed),
    ("an identical retry is served from the store", check_idempotent_retry),
    ("a different message is not cached", check_different_message_is_not_cached),
    ("a whitespace-only retry is cached", check_whitespace_retry_is_cached),
    ("a sensitive field is rejected, never filled", check_sensitive_field_is_rejected),
    ("a field with no value becomes a question", check_missing_is_asked_for),
    ("citations carry the source's title and date", check_citations_carry_title_and_date),
    ("a pending plan expires after ten minutes", check_pending_plan_lifetime),
    ("the idempotency window expires", check_idempotency_window_expires),
    ("an unknown session is a 404", check_session_not_found),
    ("runner failures map to statuses and sentences", check_failure_mapping),
    ("an unusable answer is PLAN_FAILED", check_schema_failure_is_plan_failed),
    ("a busy session is refused", check_session_busy),
    ("a field carrying a value is refused", check_value_tripwire),
    ("an over-long message is refused", check_over_long_message),
    ("the rate limit holds", check_rate_limit),
    ("an unknown page still plans", check_unsupported_page_still_plans),
    ("no content in any log line", check_logs_are_content_free),
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

    print("\nAll plan checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
