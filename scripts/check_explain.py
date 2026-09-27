"""
Check `POST /explain` without a model, a network, a database, or pytest.

    python -m scripts.check_explain

Everything runs over the real app, through the real middleware and the real exception handlers — so
what these checks read is the body a caller would actually receive, wrapped shape and all — with
four things replaced: the request's database session (a stub answering the reads this endpoint
makes), the cache table's own session (a stub that **compiles the real SQL** and keeps rows in a
dict), the model (`src/ai/fake.py`), and retrieval (a stub that records how it was called).

What it covers, against the spec's own definition of done and its error table:

 1. a sensitive field returns canned copy with no model call and no retrieval call
 2. each sensitive kind gets its own copy, and `sensitive: true` alone is enough
 3. a corpus that covers nothing says so, with no model call, and is never cached
 4. a lookup that did not run says something different, because it means something different
 5. a grounded answer carries its sources, with the corpus's own date
 6. the second click on the same field is served from the table, with no model call
 7. a cached answer survives a restart, because the cache is a table and not a dict
 8. a custom question is answered, never cached, and never stored
 9. a twice-ungrounded answer is replaced and not cached
10. an unknown, expired or foreign session answers without an agency filter, and is not an error
11. a known session narrows retrieval to its workflow's agency and service
12. every runner failure maps to the status and code in the spec's Section 5 table
13. a field carrying a value is a 400 naming the key
14. an over-long question is a 400
15. the rate limit holds
16. the cache key is stable under label formatting and changes with every part that matters
17. the cache lookup filters expiry in SQL, not in Python
18. no session row is written: no history, no chat values, no flush
19. no label, question, nearby text, explanation, example or source content in any log line
20. the cache hit rate reaches the log line

What it cannot cover, and has to be done by hand: migration `0008`, the real unique index behind
the upsert, the week-long expiry, and whether a real model's explanation of a real field reads well.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from src.agent.breaker import reset_breaker
from src.agent.constants import FAILURE_MESSAGES
from src.agent.locks import reset_locks, turn_lock
from src.agent.quota import record_usage, reset_quota
from src.ai.client import ModelPermanentError
from src.ai.fake import FakeModelClient, answer
from src.ai.prompt_loader import EXPLAIN_PROMPT_VERSION
from src.auth.schemas import AuthedUser
from src.cache import invalidate as invalidate_cache
from src.config import settings
from src.database.session import get_session
from src.documents.schemas import RetrievalResult, RetrievedChunk
from src.explain import metrics
from src.explain import scope as explain_scope
from src.explain import service as explain_service
from src.explain import store as explain_store
from src.explain.constants import (
    CANNED_EXPLANATIONS,
    GENERIC_SENSITIVE_EXPLANATION,
    NO_GUIDANCE_EXPLANATION,
    RATE_LIMIT,
    RETRIEVAL_UNAVAILABLE_EXPLANATION,
)
from src.explain.models import ExplanationCache
from src.explain.schemas import ExplainScope
from src.guard.constants import UNVERIFIED_REPLY
from src.logging import RedactingFilter, configure_logging
from src.main import app
from src.rate_limit import reset_rate_limits
from src.sessions.models import Session
from src.tokens.dependencies import require_token
from src.turn_errors import FAILURE_STATUS
from src.workflows import service as workflows_service
from src.workflows.schemas import ActiveWorkflow, Step

# --- The field, the user, the workflow and the corpus every case runs against -----------------

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_USER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
TOKEN_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
UNKNOWN_SESSION_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")

AGENCY = "National Services Portal (Demo)"
SERVICE = "Business Name Registration (Demo)"

# Planted strings. None of these may appear in any log line.
PLANTED_LABEL = "Business Type SECRETLABEL77777"
PLANTED_NEARBY = "Choose the legal form SECRETNEARBY44444"
PLANTED_QUESTION = "Does this change my fee SECRETQUESTION12345"
PLANTED_EXPLANATION = "You must pick the legal form of your business SECRETREPLY98765."
PLANTED_EXAMPLE = "Sole Proprietor SECRETEXAMPLE33333"

FIELD: dict[str, Any] = {
    "field_id": "f6",
    "label": PLANTED_LABEL,
    "type": "select",
    "required": True,
    "options": ["Sole Proprietor", "Limited Liability", "Partnership"],
}

WORKFLOWS = [
    ActiveWorkflow(id="demo_reg", agency=AGENCY, name=SERVICE, url_patterns=()),
]

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

INGESTED_AT = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)

CHUNK = RetrievedChunk(
    chunk_id=7,
    title="Demo Agency FAQ — registration",
    content="A business name may be registered as SECRETCHUNK55555 sole proprietor.",
    source_url="https://example.gov.ng/faq",
    agency=AGENCY,
    service=SERVICE,
    distance=0.2,
    ingested_at=INGESTED_AT,
)


def explain_answer(
    *,
    explanation: str = PLANTED_EXPLANATION,
    example: str | None = PLANTED_EXAMPLE,
    citations: list[dict[str, Any]] | None = None,
) -> Any:  # noqa: ANN401  # a ModelReply, from the fake's own helper
    """One scripted model answer in `ExplainResponse`'s shape."""
    return answer(
        {
            "explanation": explanation,
            "example": example,
            "citations": citations if citations is not None else [
                {"chunk_id": 7, "source_url": "https://example.gov.ng/faq"},
            ],
        },
    )


# --- The stub database ------------------------------------------------------------------------


@dataclass
class StubResult:
    """What `execute` returns: one row, or none."""

    row: Any = None

    def scalar_one_or_none(self) -> Any:  # noqa: ANN401  # a Session, or None
        return self.row


@dataclass
class StubSession:
    """
    The request's database session: the one read `/explain` makes, and a record of any write.

    Faithful where it matters. `execute` returns the session row only when the statement filters on
    the matching user id, which is what makes the "another user's session" case a check of the
    WHERE clause rather than of this stub's opinion. And it counts writes, because the property
    being checked is that there are none.
    """

    session: Session | None
    flushes: int = 0
    added: list[Any] = field(default_factory=list)
    committed: int = 0

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> StubResult:
        if self.session is None or not _matches(statement, self.session):
            return StubResult(None)

        return StubResult(self.session)

    async def flush(self) -> None:
        self.flushes += 1

    def add(self, row: Any) -> None:
        self.added.append(row)

    def add_all(self, rows: Sequence[Any]) -> None:
        self.added.extend(rows)

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        pass


def _matches(statement: Any, session: Session) -> bool:
    """
    Whether this row is the one a compiled SELECT is asking for.

    Every UUID the statement binds has to be one of the row's own. That is what makes the
    "unknown session id" and "another user's session" cases checks of the real WHERE clause — both
    ids are in it — rather than checks of this stub's opinion about which one matters.
    """
    try:
        params = statement.compile().params
    except Exception:  # noqa: BLE001  # a statement the stub does not need to understand
        return True

    bound = [value for value in params.values() if isinstance(value, uuid.UUID)]

    return all(value in (session.id, session.user_id) for value in bound)


# --- The stub cache table ---------------------------------------------------------------------


@dataclass
class CacheTable:
    """
    Rows that survive a "restart", and every statement the store compiled against Postgres.

    The store's real SQL is built and compiled here rather than mocked away: a column renamed in
    the model and not in the store would otherwise pass every check in this file and fail on the
    first request against a real database.
    """

    rows: dict[str, ExplanationCache] = field(default_factory=dict)
    statements: list[str] = field(default_factory=list)


@dataclass
class StubCacheSession:
    """One session over `CacheTable`. Compiles, then answers from the dict."""

    table: CacheTable

    async def __aenter__(self) -> "StubCacheSession":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> StubResult:
        compiled = statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
        sql = " ".join(str(compiled).split())
        self.table.statements.append(sql)

        if sql.upper().startswith("SELECT"):
            return StubResult(self.table.rows.get(_hex_param(compiled.params)))

        if sql.upper().startswith("INSERT"):
            self._insert(compiled.params)

        return StubResult(None)

    def _insert(self, params: dict[str, Any]) -> None:
        key = _hex_param(params)

        if key is None:
            return

        self.table.rows[key] = ExplanationCache(
            cache_key=key,
            explanation=params["explanation"],
            example=params["example"],
            sources=params["sources"],
            prompt_version=params["prompt_version"],
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )

    async def commit(self) -> None:
        return None


def _hex_param(params: dict[str, Any]) -> str | None:
    """The cache key in a compiled statement's parameters: the only 64-character hex value."""
    for value in params.values():
        if isinstance(value, str) and len(value) == 64:  # sha256, in hex
            return value

    return None


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
    """One request's worth of stubs: the databases, the model, retrieval, and the log."""

    db: StubSession
    model: FakeModelClient
    logs: LogCapture
    cache: CacheTable
    retrieval_calls: list[dict[str, Any]] = field(default_factory=list)


def make_session(
    *,
    user_id: uuid.UUID = USER_ID,
    workflow_id: str | None = "demo_reg",
    step_id: str | None = "demo_reg_applicant",
) -> Session:
    """An unsaved session row, as the database would have handed it back."""
    session = Session(
        user_id=user_id,
        tab_id=1,
        workflow_id=workflow_id,
        step_id=step_id,
        page_hash="hash",
        cached_rule_ids=[],
        history=[{"role": "user", "content": "earlier"}],
        chat_values={"lga": "Ikeja"},
        expires_at=datetime.now(UTC) + timedelta(hours=24),
    )
    session.id = SESSION_ID

    return session


def body(
    *,
    page_field: dict[str, Any] | None = None,
    session_id: uuid.UUID | None = SESSION_ID,
    nearby_text: str = PLANTED_NEARBY,
    question: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A well-formed request body, with room to break one thing at a time."""
    payload: dict[str, Any] = {
        "field": page_field if page_field is not None else FIELD,
        "nearby_text": nearby_text,
        **extra,
    }

    if session_id is not None:
        payload["session_id"] = str(session_id)

    if question is not None:
        payload["question"] = question

    return payload


# Distinguishes "use the default session row" from "there is no row in the database". Passing
# None for the latter is how the first version of this file quietly tested nothing.
_DEFAULT_SESSION = object()


def build_harness(
    script: Sequence[Any] = (),
    *,
    session: Any = _DEFAULT_SESSION,
    retrieval: RetrievalResult | None = None,
    cache: CacheTable | None = None,
    ingested_at: datetime | None = INGESTED_AT,
) -> tuple[TestClient, Harness]:
    """
    The app with its four edges replaced, and a client to call it with.

    Global state — the rate limiter, the turn locks, the quota, the breaker, the registry cache
    and the hit-rate counters — is reset here, so one case cannot decide the next one's outcome.
    """
    reset_rate_limits()
    reset_locks()
    reset_quota()
    reset_breaker()
    invalidate_cache()
    metrics.reset_cache_stats()

    db = StubSession(session=make_session() if session is _DEFAULT_SESSION else session)
    model = FakeModelClient(script=list(script))
    logs = LogCapture()
    table = cache if cache is not None else CacheTable()
    harness = Harness(db=db, model=model, logs=logs, cache=table)

    async def stub_retrieve(query: str, **kwargs: Any) -> RetrievalResult:
        harness.retrieval_calls.append({"query": query, **kwargs})

        if retrieval is not None:
            return retrieval

        return RetrievalResult(chunks=[CHUNK], available=True)

    async def stub_workflows(*_args: Any, **_kwargs: Any) -> list[ActiveWorkflow]:
        return WORKFLOWS

    async def stub_steps(*_args: Any, **_kwargs: Any) -> list[Step]:
        return STEPS

    async def stub_ingested_at(*_args: Any, **_kwargs: Any) -> datetime | None:
        return ingested_at

    explain_service.shared_client = lambda: model  # type: ignore[assignment]
    explain_service.search_government_information = stub_retrieve  # type: ignore[assignment]
    explain_scope.latest_ingested_at = stub_ingested_at  # type: ignore[assignment]
    # The cache opens its own session, so this is where the stub table is attached.
    explain_store.async_session_factory = lambda: StubCacheSession(table)  # type: ignore[assignment]
    workflows_service.list_active_workflows = stub_workflows  # type: ignore[assignment]
    workflows_service.get_steps = stub_steps  # type: ignore[assignment]

    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_token] = lambda: AuthedUser(
        id=USER_ID,
        email="demo@example.com",
        token_id=TOKEN_ID,
    )

    logging.getLogger().addHandler(logs)

    return TestClient(app, raise_server_exceptions=False), harness


def teardown(logs: LogCapture) -> None:
    logging.getLogger().removeHandler(logs)
    app.dependency_overrides.clear()


def expect(failures: list[str], condition: bool, description: str) -> None:  # noqa: FBT001
    if not condition:
        failures.append(description)


def data_of(response: Any) -> dict[str, Any]:  # noqa: ANN401  # httpx Response
    """The payload inside the success envelope."""
    return response.json().get("data", {})


def error_of(response: Any) -> dict[str, Any]:  # noqa: ANN401  # httpx Response
    """The payload inside the error envelope."""
    return response.json().get("error", {})


# --- Cases -----------------------------------------------------------------------------------


async def check_sensitive_field_is_canned() -> list[str]:
    """A password field: fixed copy, no model call, no lookup. The spec's second DoD item."""
    failures: list[str] = []
    client, harness = build_harness()

    try:
        sensitive = {**FIELD, "label": "Password", "type": "password", "options": []}
        response = client.post("/explain", json=body(page_field=sensitive))
        data = data_of(response)

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(
            failures,
            data.get("explanation") == CANNED_EXPLANATIONS["password"],
            "a password field should return the canned password copy",
        )
        expect(failures, data.get("grounded") is True, "canned copy is grounded")
        expect(failures, data.get("sources") == [], "canned copy cites nothing")
        expect(failures, data.get("example") is None, "canned copy carries no example")
        expect(failures, harness.model.call_count == 0, "a sensitive field made a model call")
        expect(failures, harness.retrieval_calls == [], "a sensitive field searched the corpus")
        expect(failures, harness.cache.rows == {}, "a sensitive field was cached")
        expect(
            failures,
            metrics.cache_stats().lookups == 0,
            "a sensitive field counted as a cache lookup",
        )

        # A question attached to a sensitive field must not be a way to reach the model. The
        # short-circuit is first in the sequence so nothing in the request can get past it.
        with_question = client.post(
            "/explain",
            json=body(page_field=sensitive, question="How do I get round this one?"),
        )

        expect(
            failures,
            data_of(with_question).get("explanation") == CANNED_EXPLANATIONS["password"],
            "a custom question got past the sensitive short-circuit",
        )
        expect(
            failures,
            harness.model.call_count == 0,
            "a question on a sensitive field reached the model",
        )
        expect(failures, harness.retrieval_calls == [], "a question on a sensitive field searched")
    finally:
        teardown(harness.logs)

    return failures


async def check_each_sensitive_kind() -> list[str]:
    """Every kind gets its own copy, and the content script's flag alone is enough."""
    failures: list[str] = []
    cases = [
        ("One-Time Code", "text", CANNED_EXPLANATIONS["one_time_code"]),
        ("Enter the CAPTCHA", "text", CANNED_EXPLANATIONS["captcha"]),
        ("Card Number", "text", CANNED_EXPLANATIONS["payment"]),
        ("Mother's Maiden Name", "text", GENERIC_SENSITIVE_EXPLANATION),
    ]

    for label, field_type, expected in cases:
        client, harness = build_harness()

        try:
            sensitive = {
                "field_id": "f9",
                "label": label,
                "type": field_type,
                "sensitive": True,
            }
            data = data_of(client.post("/explain", json=body(page_field=sensitive)))

            expect(failures, data.get("explanation") == expected, f"{label} got the wrong copy")
            expect(failures, harness.model.call_count == 0, f"{label} made a model call")
        finally:
            teardown(harness.logs)

    return failures


async def check_no_guidance() -> list[str]:
    """An uncovered field says so, costs no model call, and is not cached. DoD item 3."""
    failures: list[str] = []
    client, harness = build_harness(
        retrieval=RetrievalResult(chunks=[], available=True),
    )

    try:
        data = data_of(client.post("/explain", json=body()))

        expect(
            failures,
            data.get("explanation") == NO_GUIDANCE_EXPLANATION,
            "an uncovered field should get the no-guidance sentence",
        )
        expect(failures, data.get("grounded") is False, "the no-guidance answer is not grounded")
        expect(failures, data.get("sources") == [], "the no-guidance answer cites nothing")
        expect(failures, harness.model.call_count == 0, "an empty corpus still called the model")
        expect(failures, harness.cache.rows == {}, "an ungrounded answer was cached")
        expect(failures, len(harness.retrieval_calls) == 1, "expected exactly one search")
    finally:
        teardown(harness.logs)

    return failures


async def check_retrieval_unavailable_is_its_own_answer() -> list[str]:
    """A lookup that did not run must not be reported as an absence of guidance."""
    failures: list[str] = []
    client, harness = build_harness(
        retrieval=RetrievalResult(chunks=[], available=False),
    )

    try:
        data = data_of(client.post("/explain", json=body()))

        expect(
            failures,
            data.get("explanation") == RETRIEVAL_UNAVAILABLE_EXPLANATION,
            "an unavailable lookup should say it could not check",
        )
        expect(
            failures,
            data.get("explanation") != NO_GUIDANCE_EXPLANATION,
            "an unavailable lookup was reported as an empty corpus",
        )
        expect(failures, data.get("grounded") is False, "an unchecked answer is not grounded")
        expect(failures, harness.model.call_count == 0, "an unavailable lookup called the model")
        expect(failures, harness.cache.rows == {}, "an unchecked answer was cached")
    finally:
        teardown(harness.logs)

    return failures


async def check_grounded_answer() -> list[str]:
    """A real hit: one model call, the explanation, the example, and a dated source. DoD item 1."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        response = client.post("/explain", json=body())
        data = data_of(response)
        sources = data.get("sources") or []

        expect(failures, response.status_code == 200, f"status was {response.status_code}")
        expect(failures, data.get("field_id") == "f6", "the answer names the field asked about")
        expect(
            failures,
            data.get("explanation") == PLANTED_EXPLANATION,
            "the model's explanation should reach the panel",
        )
        expect(failures, data.get("example") == PLANTED_EXAMPLE, "the example was lost")
        expect(failures, data.get("grounded") is True, "a cited claim should be grounded")
        expect(failures, data.get("cached") is False, "a first answer was marked cached")
        expect(failures, len(sources) == 1, f"expected one source, got {len(sources)}")
        expect(failures, harness.model.call_count == 1, "expected exactly one model call")

        if sources:
            expect(failures, sources[0]["title"] == CHUNK.title, "the source carries its title")
            expect(failures, sources[0]["url"] == CHUNK.source_url, "the source carries its url")
            expect(
                failures,
                sources[0]["checked"] == "2026-09-20",
                "the source carries the corpus's own date, not today's",
            )

        expect(failures, len(harness.cache.rows) == 1, "a grounded answer should be cached")
        expect(
            failures,
            harness.model.calls[0].schema_name == "explain_response",
            "the explain schema should be the one enforced",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_second_click_is_cached() -> list[str]:
    """The second click on the same field: no model call, `cached: true`. DoD item 4."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        first = data_of(client.post("/explain", json=body()))
        second = data_of(client.post("/explain", json=body()))

        expect(failures, first.get("cached") is False, "the first answer was marked cached")
        expect(failures, second.get("cached") is True, "the second answer was not marked cached")
        expect(
            failures,
            second.get("explanation") == first.get("explanation"),
            "the cached answer differs from the one that was stored",
        )
        expect(
            failures,
            second.get("example") == PLANTED_EXAMPLE,
            "the cached answer lost its example",
        )
        expect(
            failures,
            (second.get("sources") or [{}])[0].get("checked") == "2026-09-20",
            "the cached answer lost its source date",
        )
        expect(failures, harness.model.call_count == 1, "the second click called the model again")
        expect(
            failures,
            len(harness.retrieval_calls) == 1,
            "the second click searched the corpus again",
        )

        stats = metrics.cache_stats()
        expect(failures, (stats.hits, stats.misses) == (1, 1), f"hit counters were {stats}")
    finally:
        teardown(harness.logs)

    return failures


async def check_cache_survives_a_restart() -> list[str]:
    """
    The reason the cache is a table.

    `fastapi dev` reloads on every save, and the demo's whole promise is that the second pass is
    instant. The rows are kept while every in-process cache and counter is thrown away.
    """
    failures: list[str] = []
    table = CacheTable()
    client, harness = build_harness([explain_answer()], cache=table)

    try:
        client.post("/explain", json=body())
    finally:
        teardown(harness.logs)

    # A "restart": new app state, new counters, new model with an empty script — so a model call
    # would raise rather than quietly answer.
    client, harness = build_harness([], cache=table)

    try:
        data = data_of(client.post("/explain", json=body()))

        expect(failures, data.get("cached") is True, "a stored answer did not survive the restart")
        expect(failures, data.get("explanation") == PLANTED_EXPLANATION, "the answer was wrong")
        expect(failures, harness.model.call_count == 0, "the restarted process called the model")
    finally:
        teardown(harness.logs)

    return failures


async def check_custom_question_is_never_cached() -> list[str]:
    """A question the user typed is answered, never looked up, and never stored."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        data = data_of(client.post("/explain", json=body(question=PLANTED_QUESTION)))

        expect(failures, data.get("cached") is False, "a custom question was served from a cache")
        expect(failures, harness.cache.rows == {}, "a custom question's answer was cached")
        expect(failures, harness.cache.statements == [], "a custom question consulted the cache")
        expect(
            failures,
            metrics.cache_stats().lookups == 0,
            "a custom question counted as a cache lookup",
        )
        expect(failures, harness.model.call_count == 1, "expected one model call")
        expect(
            failures,
            PLANTED_QUESTION in harness.retrieval_calls[0]["query"],
            "the question should shape the search query",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_ungrounded_answer_is_replaced() -> list[str]:
    """A claim with no citation: one repair, then the fixed sentence, and nothing cached."""
    failures: list[str] = []
    ungrounded = explain_answer(
        explanation="The registration fee is 5,000 naira.",
        example="₦5,000",
        citations=[],
    )
    client, harness = build_harness([ungrounded, ungrounded])

    try:
        data = data_of(client.post("/explain", json=body()))

        expect(
            failures,
            data.get("explanation") == UNVERIFIED_REPLY,
            "an unsupported claim should be replaced",
        )
        expect(failures, data.get("example") is None, "the example restating the claim survived")
        expect(failures, data.get("grounded") is False, "a replaced answer is not grounded")
        expect(failures, data.get("sources") == [], "a replaced answer cites nothing")
        expect(failures, harness.model.call_count == 2, "expected one repair, not more")
        expect(failures, harness.cache.rows == {}, "an ungrounded answer was cached")
    finally:
        teardown(harness.logs)

    return failures


async def check_unknown_session_still_answers() -> list[str]:
    """An unknown, expired or foreign session is a 200 with no agency filter, not an error."""
    failures: list[str] = []
    cases = [
        ("no session id at all", None, make_session()),
        ("an unknown session id", UNKNOWN_SESSION_ID, make_session()),
        ("an expired or deleted session", SESSION_ID, None),
        ("another user's session", SESSION_ID, make_session(user_id=OTHER_USER_ID)),
    ]

    for description, session_id, row in cases:
        client, harness = build_harness([explain_answer()], session=row)

        try:
            response = client.post(
                "/explain",
                json=body(session_id=session_id),
            )
            data = data_of(response)

            expect(
                failures,
                response.status_code == 200,
                f"{description} gave {response.status_code}, not 200",
            )
            expect(failures, bool(data.get("explanation")), f"{description} produced no answer")
            expect(
                failures,
                harness.retrieval_calls[0]["agency"] is None,
                f"{description} still filtered retrieval by agency",
            )
            expect(
                failures,
                harness.retrieval_calls[0]["service"] is None,
                f"{description} still filtered retrieval by service",
            )
        finally:
            teardown(harness.logs)

    return failures


async def check_known_session_narrows_retrieval() -> list[str]:
    """A live session scopes the search to its workflow's agency and service."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        client.post("/explain", json=body())
        call = harness.retrieval_calls[0]

        expect(failures, call["agency"] == AGENCY, f"agency filter was {call['agency']!r}")
        expect(failures, call["service"] == SERVICE, f"service filter was {call['service']!r}")
        expect(failures, call["limit"] == 3, f"search limit was {call['limit']}")
        expect(
            failures,
            "Applicant Information" in harness.model.calls[0].text_at(1),
            "the prompt should carry the step line from the session's workflow",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_failure_mapping() -> list[str]:
    """Every runner failure maps to the status, code and sentence of the spec's Section 5 table."""
    failures: list[str] = []

    # One script per failure, and each one is the shortest way to produce it. `SCHEMA_FAILED`
    # needs three unusable replies: the first while tools are still offered, the final call after
    # they are withdrawn, and the one repair.
    cases: list[tuple[str, str, list[Any]]] = [
        ("MODEL_UNAVAILABLE", "MODEL_UNAVAILABLE", [ModelPermanentError("unavailable")]),
        ("SCHEMA_FAILED", "PLAN_FAILED", [answer("not json at all")] * 3),
        ("BUDGET_EXCEEDED", "MODEL_TIMEOUT", []),
    ]

    for code, expected_code, script in cases:
        client, harness = build_harness(script)

        if code == "BUDGET_EXCEEDED":
            # The deadline is gone before the first call, so no reply is ever needed.
            explain_service.REQUEST_BUDGET_SECONDS = 0.0  # type: ignore[assignment]

        try:
            response = client.post("/explain", json=body())
            error = error_of(response)

            expect(
                failures,
                response.status_code == FAILURE_STATUS[code],
                f"{code} gave {response.status_code}, expected {FAILURE_STATUS[code]}",
            )
            expect(
                failures,
                error.get("code") == expected_code,
                f"{code} returned code {error.get('code')!r}",
            )
            expect(
                failures,
                error.get("message") == FAILURE_MESSAGES[code],  # type: ignore[index]
                f"{code} did not carry the runner's own sentence",
            )
        finally:
            explain_service.REQUEST_BUDGET_SECONDS = _REQUEST_BUDGET  # type: ignore[assignment]
            teardown(harness.logs)

    return failures


async def check_quota_is_refused() -> list[str]:
    """A user over the daily ceiling is told so, as a 429, before any call is made."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        record_usage(str(USER_ID), settings.daily_token_quota + 1)
        response = client.post("/explain", json=body())
        error = error_of(response)

        expect(failures, response.status_code == 429, f"status was {response.status_code}")
        expect(failures, error.get("code") == "QUOTA_EXCEEDED", f"code was {error.get('code')!r}")
        expect(failures, harness.model.call_count == 0, "a user over quota still called the model")
    finally:
        reset_quota()
        teardown(harness.logs)

    return failures


async def check_explain_does_not_share_the_plan_lock() -> list[str]:
    """
    Two clicks on one field are refused; a plan turn in flight is not allowed to refuse them.

    The second half is the point of `_turn_key`. `/explain` writes nothing and races nothing, so a
    plan turn holding the session's slot must not be able to stop the user reading about a field —
    which is what would happen if both locked on the bare session id.
    """
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        async with turn_lock(f"explain:{SESSION_ID}:f6"):
            response = client.post("/explain", json=body())

        expect(failures, response.status_code == 409, f"status was {response.status_code}")
        expect(failures, error_of(response).get("code") == "SESSION_BUSY", "wrong code")
        expect(failures, harness.model.call_count == 0, "a busy field still called the model")

        # A plan turn for the same session holds the bare session id. Explain goes ahead.
        async with turn_lock(str(SESSION_ID)):
            response = client.post("/explain", json=body())

        expect(
            failures,
            response.status_code == 200,
            f"a plan turn in flight blocked /explain ({response.status_code})",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_value_tripwire() -> list[str]:
    """A field object carrying what the user typed is refused, by name, without echoing it."""
    failures: list[str] = []
    client, harness = build_harness()

    try:
        poisoned = {**FIELD, "value": "hunter2"}
        response = client.post("/explain", json=body(page_field=poisoned))
        error = error_of(response)

        expect(failures, response.status_code == 400, f"status was {response.status_code}")
        expect(failures, error.get("code") == "INVALID_REQUEST", "wrong error code")
        expect(failures, "value" in error.get("message", ""), "the message should name the key")
        expect(
            failures,
            "hunter2" not in response.text,
            "the rejected value was echoed back to the caller",
        )
        expect(failures, "hunter2" not in harness.logs.text, "the value reached a log line")
    finally:
        teardown(harness.logs)

    return failures


async def check_over_long_question() -> list[str]:
    """A question past its cap is refused rather than truncated."""
    failures: list[str] = []
    client, harness = build_harness()

    try:
        response = client.post("/explain", json=body(question="x" * 501))

        expect(failures, response.status_code == 400, f"status was {response.status_code}")
        expect(failures, harness.model.call_count == 0, "an invalid request reached the model")

        response = client.post("/explain", json=body(nearby_text="y" * 301))

        expect(
            failures,
            response.status_code == 400,
            f"over-long nearby text gave {response.status_code}",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_rate_limit() -> list[str]:
    """Thirty a minute per token, then a 429 carrying the wait."""
    failures: list[str] = []
    client, harness = build_harness()

    try:
        sensitive = {**FIELD, "label": "Password", "type": "password", "options": []}
        statuses = [
            client.post("/explain", json=body(page_field=sensitive)).status_code
            for _ in range(RATE_LIMIT + 1)
        ]

        expect(
            failures,
            statuses[:RATE_LIMIT] == [200] * RATE_LIMIT,
            "a request inside the limit was refused",
        )
        expect(failures, statuses[-1] == 429, f"the request past the limit gave {statuses[-1]}")

        response = client.post("/explain", json=body(page_field=sensitive))
        error = error_of(response)

        expect(failures, error.get("code") == "RATE_LIMITED", "wrong code past the limit")
        expect(failures, error.get("retry_after", 0) > 0, "no retry_after on the 429")
    finally:
        teardown(harness.logs)

    return failures


async def check_cache_key() -> list[str]:
    """The key is stable under formatting and changes with every part that matters."""
    failures: list[str] = []
    scope = ExplainScope(
        workflow_id="demo_reg",
        step_id="demo_reg_applicant",
        agency=AGENCY,
        service=SERVICE,
    )
    parts = {"prompt_version": EXPLAIN_PROMPT_VERSION, "corpus_stamp": INGESTED_AT.isoformat()}

    base = explain_store.cache_key_for(scope, "Business Type", has_question=False, **parts)
    formatted = explain_store.cache_key_for(scope, " business type * ", has_question=False, **parts)

    expect(failures, base == formatted, "a label's formatting changed its key")
    expect(failures, base is not None and len(base) == 64, "the key should be a sha256 hex digest")

    others = {
        "a different label": explain_store.cache_key_for(
            scope, "Business Address", has_question=False, **parts,
        ),
        "a different step": explain_store.cache_key_for(
            ExplainScope(workflow_id="demo_reg", step_id="demo_reg_verify"),
            "Business Type",
            has_question=False,
            **parts,
        ),
        "a different workflow": explain_store.cache_key_for(
            ExplainScope(workflow_id="other", step_id="demo_reg_applicant"),
            "Business Type",
            has_question=False,
            **parts,
        ),
        "no workflow at all": explain_store.cache_key_for(
            ExplainScope(), "Business Type", has_question=False, **parts,
        ),
        "a new prompt version": explain_store.cache_key_for(
            scope,
            "Business Type",
            has_question=False,
            prompt_version="v99",
            corpus_stamp=parts["corpus_stamp"],
        ),
        "a re-ingested corpus": explain_store.cache_key_for(
            scope,
            "Business Type",
            has_question=False,
            prompt_version=EXPLAIN_PROMPT_VERSION,
            corpus_stamp="2026-10-01T00:00:00+00:00",
        ),
    }

    for description, key in others.items():
        expect(failures, key != base, f"{description} produced the same key")

    expect(
        failures,
        explain_store.cache_key_for(scope, "Business Type", has_question=True, **parts) is None,
        "a custom question must have no cache key at all",
    )
    expect(
        failures,
        explain_store.cache_key_for(scope, "  ", has_question=False, **parts) is None,
        "an unlabelled field must have no cache key: nothing in it tells one control from another",
    )

    return failures


async def check_expiry_is_filtered_in_sql() -> list[str]:
    """
    The cache's expiry is a WHERE clause, not a Python comparison.

    A row past its week must be unreachable even if the cleanup sweep has never run, and that is
    only true if the query says so — which is a property of the SQL, checkable without a database.
    """
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        client.post("/explain", json=body())
        selects = [sql for sql in harness.cache.statements if sql.upper().startswith("SELECT")]
        inserts = [sql for sql in harness.cache.statements if sql.upper().startswith("INSERT")]

        expect(failures, len(selects) == 1, f"expected one cache SELECT, got {len(selects)}")
        expect(
            failures,
            bool(selects) and "expires_at > now()" in selects[0],
            "the cache lookup does not filter expiry in SQL",
        )
        expect(failures, len(inserts) == 1, f"expected one cache INSERT, got {len(inserts)}")
        expect(
            failures,
            bool(inserts) and "ON CONFLICT" in inserts[0].upper(),
            "the cache write is not an upsert",
        )
    finally:
        teardown(harness.logs)

    return failures


async def check_nothing_is_written_to_the_session() -> list[str]:
    """`/explain` is a side call: no history, no chat values, no flush, no commit of its own."""
    failures: list[str] = []
    session = make_session()
    history_before = list(session.history)
    chat_before = dict(session.chat_values)
    client, harness = build_harness([explain_answer()], session=session)

    try:
        client.post("/explain", json=body(question=PLANTED_QUESTION))

        expect(failures, harness.db.flushes == 0, "the session was flushed")
        expect(failures, harness.db.added == [], "a row was added to the request's session")
        expect(failures, session.history == history_before, "the session's history changed")
        expect(failures, session.chat_values == chat_before, "the session's chat values changed")
    finally:
        teardown(harness.logs)

    return failures


async def check_logs_are_content_free() -> list[str]:
    """Not a label, a question, a nearby sentence, an explanation, an example or a source."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        client.post("/explain", json=body(question=PLANTED_QUESTION))
        text = harness.logs.text

        forbidden = {
            "the field label": "SECRETLABEL77777",
            "the nearby text": "SECRETNEARBY44444",
            "the user's question": "SECRETQUESTION12345",
            "the explanation": "SECRETREPLY98765",
            "the example": "SECRETEXAMPLE33333",
            "retrieved content": "SECRETCHUNK55555",
        }

        for description, planted in forbidden.items():
            expect(failures, planted not in text, f"{description} appeared in a log line")

        expect(failures, "explain user=" in text, "the request line was not emitted")
        expect(failures, "question_chars=" in text, "the request line omits the question length")
        expect(failures, "cache_hit_rate=" in text, "the request line omits the cache hit rate")
        expect(failures, "turn kind=explain" in text, "the runner's turn line was not emitted")
    finally:
        teardown(harness.logs)

    return failures


async def check_hit_rate_reaches_the_log() -> list[str]:
    """DoD item 5: the rate climbs as the second pass over a form is served from the table."""
    failures: list[str] = []
    client, harness = build_harness([explain_answer()])

    try:
        client.post("/explain", json=body())
        client.post("/explain", json=body())

        expect(
            failures,
            "cache_hits=0/1 cache_hit_rate=0.00" in harness.logs.text,
            "the first request should log a miss",
        )
        expect(
            failures,
            "cache_hits=1/2 cache_hit_rate=0.50" in harness.logs.text,
            "the second request should log a hit",
        )
    finally:
        teardown(harness.logs)

    return failures


_REQUEST_BUDGET = explain_service.REQUEST_BUDGET_SECONDS


CHECKS: list[tuple[str, Any]] = [
    ("a sensitive field is canned, with no model call", check_sensitive_field_is_canned),
    ("each sensitive kind gets its own copy", check_each_sensitive_kind),
    ("an uncovered field says so, for free", check_no_guidance),
    ("a lookup that did not run is its own answer", check_retrieval_unavailable_is_its_own_answer),
    ("a grounded answer carries a dated source", check_grounded_answer),
    ("the second click is served from the table", check_second_click_is_cached),
    ("the cache survives a restart", check_cache_survives_a_restart),
    ("a custom question is never cached", check_custom_question_is_never_cached),
    ("an unsupported claim is replaced, not cached", check_ungrounded_answer_is_replaced),
    ("an unknown session still answers", check_unknown_session_still_answers),
    ("a known session narrows retrieval", check_known_session_narrows_retrieval),
    ("runner failures map to statuses and sentences", check_failure_mapping),
    ("a user over quota is refused", check_quota_is_refused),
    ("explain does not share the plan lock", check_explain_does_not_share_the_plan_lock),
    ("a field carrying a value is refused", check_value_tripwire),
    ("an over-long question is refused", check_over_long_question),
    ("the rate limit holds", check_rate_limit),
    ("the cache key is stable and specific", check_cache_key),
    ("the cache filters expiry in SQL", check_expiry_is_filtered_in_sql),
    ("nothing is written to the session", check_nothing_is_written_to_the_session),
    ("no content in any log line", check_logs_are_content_free),
    ("the cache hit rate reaches the log", check_hit_rate_reaches_the_log),
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

    print("\nAll explain checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
