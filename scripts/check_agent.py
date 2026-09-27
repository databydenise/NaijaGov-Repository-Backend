"""
Check the agent runner without a model, a network, a database, or pytest.

    python -m scripts.check_agent

Everything runs against `src/ai/fake.py`, a model that replays a scripted sequence of replies.
That is the only way to test what this fragment is actually about — the caps and the failure
paths are *sequences*, and a real model does not produce a chosen sequence on demand.

What this covers, against P4's definition of done:

1. Every failure mode returns a code and a sentence a person can read (`failure catalogue`,
   and one case per code).
2. No path exceeds the caps in Section 5: two tool rounds, five retrieval calls, two transient
   retries, one repair pass, one turn budget — each asserted by counting what the fake saw.
3. The whole file runs offline. No API key is read, no client is constructed, nothing resolves
   a hostname.
4. The telemetry is content-free: a turn is run with planted strings and neither the record nor
   any log line it emits contains one.

What it cannot cover, and what has to be done by hand: whether the provider accepts
`PLAN_RESPONSE_SCHEMA` under `strict: true`, and whether the prompts produce good behaviour.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.agent import tools as agent_tools
from src.agent.breaker import reset_breaker
from src.agent.constants import (
    FAILURE_MESSAGES,
    MAX_RETRIEVAL_CALLS,
    MAX_TOOL_ROUNDS,
    MAX_TRANSIENT_RETRIES,
    FailureCode,
)
from src.agent.explain import run_explain_turn
from src.agent.locks import reset_locks
from src.agent.plan import run_plan_turn
from src.agent.quota import record_usage, reset_quota
from src.agent.schemas import ExplainTurn, PlanTurn, TurnFailure
from src.ai.client import ModelPermanentError, ModelTimeoutError, ModelTransportError
from src.ai.context import PlanContext
from src.ai.fake import FakeModelClient, answer, searches
from src.config import settings
from src.context.schemas import PageButton, PageField
from src.documents.schemas import RetrievalResult, RetrievedChunk
from src.guard.constants import UNVERIFIED_REPLY

# --- The page and the user every case runs against --------------------------------------------

FIELDS = [
    PageField(field_id="f_name", label="Full Name", type="text", required=True),
    PageField(field_id="f_email", label="Email Address", type="email", required=True),
    PageField(field_id="f_state", label="State", type="select", required=True,
              options=["Lagos", "Kano", "Rivers"]),
    PageField(field_id="f_pw", label="Password", type="password", required=True, sensitive=True),
]
BUTTONS = [
    PageButton(field_id="b_next", text="Continue"),
    PageButton(field_id="b_submit", text="Submit Application", sensitive=True),
]
PROFILE: dict[str, str | None] = {
    "full_name": "Adaeze Okonkwo",
    "email": "adaeze@example.com",
    "state": "Lagos",
}
CHAT = {"lga": "Ikeja"}

CONTEXT = PlanContext(text="<page_snapshot>…</page_snapshot>", estimated_tokens=900, truncated=[])

CHUNK = RetrievedChunk(
    chunk_id=7,
    title="Fees",
    content="The registration fee is ₦5,000.",
    source_url="https://example.gov.ng/fees",
    agency="Demo Agency",
    service="Demo Service",
    distance=0.2,
)

PLAIN_REPLY = "I filled in what I had. Have a look before you continue."
FEE_REPLY = "The registration fee is ₦5,000 and processing takes 14 working days."


def plan_payload(
    *,
    reply: str = PLAIN_REPLY,
    actions: Sequence[dict[str, Any]] = (),
    citations: Sequence[dict[str, Any]] = (),
    missing: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """A response body the schema accepts. `extracted_data` is required, so it is always sent."""
    return {
        "reply": reply,
        "actions": list(actions),
        "extracted_data": {},
        "citations": list(citations),
        "missing": list(missing),
    }


def fill(field_id: str, key: str, source: str = "profile") -> dict[str, Any]:
    return {
        "type": "fill",
        "field_id": field_id,
        "value_ref": {"source": source, "key": key},
    }


GOOD_PLAN = plan_payload(actions=[fill("f_email", "email")])


# --- Stubs ------------------------------------------------------------------------------------


@dataclass
class RetrievalStub:
    """A retrieval callable that records what it was asked, and can fail on demand."""

    chunks: Sequence[RetrievedChunk] = ()
    available: bool = True
    delay: float = 0.0
    error: Exception | None = None
    queries: list[str] = None  # type: ignore[assignment]  # set in __post_init__

    def __post_init__(self) -> None:
        self.queries = []

    async def __call__(
        self,
        query: str,
        *,
        limit: int = 3,
        agency: str | None = None,
        service: str | None = None,
    ) -> RetrievalResult:
        self.queries.append(query)

        if self.delay:
            await asyncio.sleep(self.delay)

        if self.error is not None:
            raise self.error

        return RetrievalResult(chunks=list(self.chunks), available=self.available)


async def run_plan(
    script: Sequence[Any],
    *,
    retrieve: RetrievalStub | None = None,
    session_id: str = "s1",
    user_id: str = "u1",
    deadline: float | None = None,
    fields: Sequence[PageField] = tuple(FIELDS),
) -> tuple[PlanTurn | TurnFailure, FakeModelClient, RetrievalStub]:
    """One plan turn against a scripted model. State is reset first, never between checks."""
    reset_locks()
    reset_quota()
    reset_breaker()

    client = FakeModelClient(list(script))
    stub = retrieve if retrieve is not None else RetrievalStub()

    outcome = await run_plan_turn(
        context=CONTEXT,
        fields=fields,
        buttons=BUTTONS,
        profile=PROFILE,
        chat_values=CHAT,
        retrieve=stub,
        client=client,
        session_id=session_id,
        user_id=user_id,
        deadline=deadline,
    )

    return outcome, client, stub


def expect(failures: list[str], condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        failures.append(message)


CHECKS: list[tuple[str, Callable[[], Awaitable[list[str]]]]] = []


def check(name: str) -> Callable[[Callable[[], Awaitable[list[str]]]], Any]:
    """Register one named check. Each returns the list of things that were not true."""

    def register(func: Callable[[], Awaitable[list[str]]]) -> Callable[[], Awaitable[list[str]]]:
        CHECKS.append((name, func))

        return func

    return register


# --- The happy paths ---------------------------------------------------------------------------


@check("one call, no tools, a guarded plan")
async def _happy_path() -> list[str]:
    failures: list[str] = []
    outcome, client, stub = await run_plan([answer(GOOD_PLAN)])

    expect(failures, isinstance(outcome, PlanTurn), "a scripted good answer should give a plan")

    if not isinstance(outcome, PlanTurn):
        return failures

    expect(failures, client.call_count == 1, f"expected one model call, got {client.call_count}")
    expect(failures, client.calls[0].tools_offered, "the first call must offer the search tool")
    expect(failures, client.calls[0].schema_enforced, "every call must enforce the schema")
    expect(failures, len(outcome.plan.approved) == 1, "the email fill should be approved")
    expect(
        failures,
        outcome.plan.approved[0].value == PROFILE["email"],
        "the guard should substitute the real value",
    )
    expect(failures, not stub.queries, "nothing should be searched when the model does not ask")
    expect(failures, outcome.telemetry.outcome == "ok", "telemetry outcome should be ok")
    expect(failures, outcome.telemetry.model_calls == 1, "telemetry should count one model call")
    expect(
        failures,
        outcome.telemetry.total_tokens > 0,
        "telemetry should carry the token counts the cost is estimated from",
    )

    return failures


@check("a search grounds a claim and its citation survives")
async def _grounded() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    script = [
        searches("what is the registration fee"),
        answer(
            plan_payload(
                reply=FEE_REPLY,
                citations=[{"chunk_id": 7, "source_url": "https://example.gov.ng/fees"}],
            ),
        ),
    ]
    outcome, client, stub = await run_plan(script, retrieve=stub)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, len(stub.queries) == 1, "the search should have run once")
    expect(failures, outcome.grounding == "grounded", f"expected grounded, got {outcome.grounding}")
    expect(failures, len(outcome.chunks) == 1, "the retrieved chunk should reach the caller")
    expect(failures, len(outcome.plan.citations) == 1, "the citation should verify")
    expect(failures, outcome.telemetry.tool_calls == 1, "telemetry should count the tool call")
    expect(failures, outcome.telemetry.chunks_retrieved == 1, "telemetry should count the chunk")
    expect(
        failures,
        client.calls[1].roles == ("system", "user", "assistant", "tool"),
        f"the second call should replay the tool exchange, got {client.calls[1].roles}",
    )

    return failures


@check("a claim with nothing retrieved is unverified and replaced")
async def _ungrounded() -> list[str]:
    failures: list[str] = []
    # Two ungrounded answers: the first earns the repair, the second exhausts it.
    script = [answer(plan_payload(reply=FEE_REPLY)), answer(plan_payload(reply=FEE_REPLY))]
    outcome, client, _ = await run_plan(script)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, client.call_count == 2, f"expected a repair call, got {client.call_count}")
    expect(failures, outcome.telemetry.repair_ran, "telemetry should record the repair")
    expect(
        failures,
        outcome.telemetry.repair_reason == "unverified",
        f"repair reason should be unverified, got {outcome.telemetry.repair_reason}",
    )
    expect(
        failures,
        outcome.plan.reply == UNVERIFIED_REPLY,
        "a twice-ungrounded reply must be replaced with the fixed sentence",
    )
    expect(failures, not outcome.plan.citations, "nothing cited means nothing shown")

    return failures


@check("the repair call carries the context, not the failed answer")
async def _repair_shape() -> list[str]:
    failures: list[str] = []
    script = [answer(plan_payload(reply=FEE_REPLY)), answer(plan_payload(reply=PLAIN_REPLY))]
    outcome, client, _ = await run_plan(script)

    if client.call_count != 2:
        return [f"expected two calls, got {client.call_count}"]

    repair = client.calls[1]

    expect(
        failures,
        repair.roles == ("system", "user", "user"),
        f"the repair should be context plus instruction, got {repair.roles}",
    )
    expect(failures, not repair.tools_offered, "the repair call must not offer tools")
    expect(failures, repair.schema_enforced, "the repair call must enforce the schema")
    expect(
        failures,
        FEE_REPLY not in repair.text_at(2),
        "the failed answer must not be fed back to the model",
    )
    expect(
        failures,
        isinstance(outcome, PlanTurn) and outcome.plan.reply == PLAIN_REPLY,
        "a repaired reply should be the one returned",
    )

    return failures


@check("a legitimate rejection does not earn a repair")
async def _no_repair_for_sensitive() -> list[str]:
    failures: list[str] = []
    script = [answer(plan_payload(actions=[fill("f_pw", "email")]))]
    outcome, client, _ = await run_plan(script)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, client.call_count == 1, "a sensitive-field rejection must not be re-asked")
    expect(failures, not outcome.telemetry.repair_ran, "no repair should have run")
    expect(failures, outcome.plan.rejected[0].code == "SENSITIVE_FIELD", "expected SENSITIVE_FIELD")
    expect(failures, outcome.plan.rejected[0].message, "a rejection must carry a sentence")

    return failures


@check("a sensitive field the model reported as missing is dropped")
async def _never_ask_for_a_password() -> list[str]:
    failures: list[str] = []
    # Found by a live turn, not by a check: the model put the Password field in its own `missing`
    # list and the guard passed it through, because the sensitivity filter only covered the
    # entries the guard adds itself. "What is your Password?" is the one question this product
    # must never ask.
    script = [
        answer(
            plan_payload(
                actions=[fill("f_email", "email")],
                missing=[
                    {
                        "field_id": "f_pw",
                        "label": "Password",
                        "question": "Please enter your password.",
                    },
                    {"field_id": "f_state", "label": "State", "question": "Which state?"},
                ],
            ),
        ),
    ]
    outcome, _, _ = await run_plan(script)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    asked_for = {item.field_id for item in outcome.plan.missing}

    expect(failures, "f_pw" not in asked_for, "the guard must never ask for a sensitive field")
    expect(failures, "f_state" in asked_for, "an ordinary reported gap should survive")

    return failures


@check("a schema failure names the rule it broke, without content")
async def _schema_error_signatures() -> list[str]:
    failures: list[str] = []
    # A `fill` with no `value_ref` passes the provider's strict schema (the field is nullable) and
    # fails ours (a write must name a reference). This is the shape a live turn most likely hit.
    broken = {
        "reply": PLAIN_REPLY,
        "actions": [{"type": "fill", "field_id": "f_email"}],
        "extracted_data": {},
        "citations": [],
        "missing": [],
    }
    outcome, _, _ = await run_plan([answer(broken), answer(GOOD_PLAN)])

    if not isinstance(outcome, PlanTurn):
        return [f"expected a repaired plan, got {outcome.code}"]

    signatures = outcome.telemetry.schema_errors

    expect(failures, bool(signatures), "the telemetry should name the failure")
    expect(
        failures,
        any("actions" in signature for signature in signatures),
        f"the signature should locate the action, got {signatures}",
    )
    expect(
        failures,
        all(":" in signature for signature in signatures),
        "each signature should be location:type",
    )
    expect(
        failures,
        not any("f_email" in signature or PLAIN_REPLY in signature for signature in signatures),
        "a signature must not carry a field id's value or the reply",
    )

    return failures


@check("actions rejected as malformed do earn the repair")
async def _repair_for_malformed() -> list[str]:
    failures: list[str] = []
    # A `fill` aimed at a <select> is malformed, never coerced.
    bad = plan_payload(actions=[fill("f_state", "state")])
    script = [answer(bad), answer(GOOD_PLAN)]
    outcome, client, _ = await run_plan(script)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, client.call_count == 2, f"expected a repair call, got {client.call_count}")
    expect(
        failures,
        outcome.telemetry.repair_reason == "malformed_actions",
        f"expected malformed_actions, got {outcome.telemetry.repair_reason}",
    )
    expect(failures, len(outcome.plan.approved) == 1, "the repaired action should be approved")

    return failures


@check("a mid-sequence reply that is not the answer costs a call, not the repair")
async def _mid_sequence_discard() -> list[str]:
    failures: list[str] = []
    # Found live: with tools still on offer the model may narrate what it read instead of
    # answering, and that prose is not a failed answer. The tools come off and it is asked again —
    # the repair pass stays in reserve for a final call that really does break the schema.
    outcome, client, _ = await run_plan([answer("here is what I found: …"), answer(GOOD_PLAN)])

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, client.call_count == 2, f"expected two calls, got {client.call_count}")
    expect(failures, client.calls[0].tools_offered, "the discarded call had tools on offer")
    expect(failures, not client.calls[1].tools_offered, "the next call must withdraw them")
    expect(failures, not outcome.telemetry.repair_ran, "the repair pass must not have been spent")
    expect(
        failures,
        outcome.telemetry.discarded_replies == 1,
        f"the discard should be counted, got {outcome.telemetry.discarded_replies}",
    )
    expect(
        failures,
        any("json_invalid" in signature for signature in outcome.telemetry.schema_errors),
        f"the reason should still be recorded, got {outcome.telemetry.schema_errors}",
    )

    return failures


@check("an unparseable final answer is repaired once, then SCHEMA_FAILED")
async def _schema_repair() -> list[str]:
    failures: list[str] = []
    # Three replies: one discarded mid-sequence, one that fails as the final answer and spends the
    # repair, and the repaired answer itself.
    outcome, client, _ = await run_plan(
        [answer("narration"), answer("not json at all"), answer(GOOD_PLAN)],
    )

    expect(failures, isinstance(outcome, PlanTurn), "a repaired answer should give a plan")
    expect(failures, client.call_count == 3, f"expected three calls, got {client.call_count}")

    if isinstance(outcome, PlanTurn):
        expect(
            failures,
            outcome.telemetry.repair_reason == "schema",
            f"expected a schema repair, got {outcome.telemetry.repair_reason}",
        )
        expect(failures, not client.calls[2].tools_offered, "the repair call offers no tools")

    failed, client2, _ = await run_plan(
        [answer("narration"), answer("nope"), answer("still nope")],
    )

    expect(failures, isinstance(failed, TurnFailure), "a bad answer twice should fail the turn")
    expect(
        failures,
        isinstance(failed, TurnFailure) and failed.code == "SCHEMA_FAILED",
        "the code should be SCHEMA_FAILED",
    )
    expect(failures, client2.call_count == 3, "exactly one repair attempt, then stop")

    return failures


@check("one repair per turn, whatever diagnosed it")
async def _single_repair() -> list[str]:
    failures: list[str] = []
    # A schema failure on the final answer spends the repair; the ungrounded answer it produces
    # cannot spend it again, so the claim is replaced instead of re-asked. The first reply is
    # discarded mid-sequence, which deliberately costs a call and nothing else.
    script = [
        answer("narration"),
        answer("not json"),
        answer(plan_payload(reply=FEE_REPLY)),
    ]
    outcome, client, _ = await run_plan(script)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    expect(failures, client.call_count == 3, f"expected three calls in all, got {client.call_count}")
    expect(
        failures,
        outcome.plan.reply == UNVERIFIED_REPLY,
        "with the repair already spent, the unsupported claim must be replaced",
    )

    return failures


# --- The caps ------------------------------------------------------------------------------------


@check(f"at most {MAX_TOOL_ROUNDS} tool rounds, then tools are withdrawn")
async def _tool_rounds() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    # The model asks to search on every reply it is allowed to.
    script = [searches("one"), searches("two"), answer(GOOD_PLAN)]
    outcome, client, stub = await run_plan(script, retrieve=stub)

    expect(failures, isinstance(outcome, PlanTurn), "the turn should still finish")
    expect(failures, client.call_count == 3, f"expected three model calls, got {client.call_count}")
    expect(failures, client.calls[0].tools_offered, "round one offers tools")
    expect(failures, client.calls[1].tools_offered, "round two offers tools")
    expect(failures, not client.calls[2].tools_offered, "the final call must offer no tools")
    expect(failures, len(stub.queries) == 2, f"expected two searches, got {len(stub.queries)}")

    return failures


@check(f"at most {MAX_RETRIEVAL_CALLS} retrieval calls per turn")
async def _retrieval_cap() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    # Four distinct queries, then four more: eight asked for, five may be served.
    script = [
        searches("a", "b", "c", "d"),
        searches("e", "f", "g", "h"),
        answer(GOOD_PLAN),
    ]
    outcome, client, stub = await run_plan(script, retrieve=stub)

    expect(failures, isinstance(outcome, PlanTurn), "the turn should still finish")
    expect(
        failures,
        len(stub.queries) <= MAX_RETRIEVAL_CALLS,
        f"{len(stub.queries)} searches ran, cap is {MAX_RETRIEVAL_CALLS}",
    )

    if isinstance(outcome, PlanTurn):
        expect(
            failures,
            outcome.telemetry.tool_calls == 8,
            f"all eight tool calls should be counted, got {outcome.telemetry.tool_calls}",
        )

    # Every refused call is still answered, or the next model call is malformed. The last call
    # carries the whole conversation, so counting there counts each reply once.
    tool_messages = [
        message for message in client.calls[-1].messages if message.get("role") == "tool"
    ]
    expect(failures, len(tool_messages) == 8, f"expected eight tool replies, got {len(tool_messages)}")

    refusals = [
        message for message in tool_messages if "Search limit reached" in str(message["content"])
    ]
    expect(failures, len(refusals) == 3, f"expected three limit messages, got {len(refusals)}")

    return failures


@check("an identical query is served from the turn cache")
async def _query_cache() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    script = [searches("renewal fee", "Renewal Fee", "renewal fee"), answer(GOOD_PLAN)]
    outcome, _, stub = await run_plan(script, retrieve=stub)

    expect(failures, isinstance(outcome, PlanTurn), "the turn should finish")
    expect(
        failures,
        len(stub.queries) == 1,
        f"the same query should be executed once, ran {len(stub.queries)} times",
    )

    return failures


@check(f"at most {MAX_TRANSIENT_RETRIES} transient retries, then MODEL_UNAVAILABLE")
async def _retry_cap() -> list[str]:
    failures: list[str] = []
    boom = ModelTransportError("down")
    outcome, client, _ = await run_plan([boom, boom, boom, boom])

    expect(failures, isinstance(outcome, TurnFailure), "a dead provider should fail the turn")
    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "MODEL_UNAVAILABLE",
        "the code should be MODEL_UNAVAILABLE",
    )
    expect(
        failures,
        client.call_count == MAX_TRANSIENT_RETRIES + 1,
        f"expected {MAX_TRANSIENT_RETRIES + 1} attempts, got {client.call_count}",
    )
    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.telemetry.retries == MAX_TRANSIENT_RETRIES,
        "telemetry should count the retries",
    )

    return failures


@check("a timeout is retried, then reported as MODEL_TIMEOUT")
async def _timeout() -> list[str]:
    failures: list[str] = []
    late = ModelTimeoutError("slow")
    outcome, client, _ = await run_plan([late, late, late])

    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "MODEL_TIMEOUT",
        "three timeouts should report MODEL_TIMEOUT",
    )
    expect(failures, client.call_count == 3, f"expected three attempts, got {client.call_count}")

    # A recovered call must not be reported as a failure.
    ok, client2, _ = await run_plan([late, answer(GOOD_PLAN)])
    expect(failures, isinstance(ok, PlanTurn), "a retry that succeeds should give a plan")
    expect(failures, client2.call_count == 2, "one retry, one success")

    return failures


@check("a permanent failure is not retried")
async def _permanent() -> list[str]:
    failures: list[str] = []
    outcome, client, _ = await run_plan([ModelPermanentError("no key")])

    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "MODEL_UNAVAILABLE",
        "a missing key should be MODEL_UNAVAILABLE",
    )
    expect(failures, client.call_count == 1, "a permanent failure must not be re-sent")

    return failures


@check("an exhausted budget starts nothing")
async def _budget() -> list[str]:
    failures: list[str] = []
    outcome, client, _ = await run_plan([answer(GOOD_PLAN)], deadline=time.monotonic() - 1)

    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "BUDGET_EXCEEDED",
        "a passed deadline should be BUDGET_EXCEEDED",
    )
    expect(failures, client.call_count == 0, "no call should be made without budget for it")

    return failures


@check("a turn near its deadline does not start a search")
async def _retrieval_budget() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    script = [searches("fees"), answer(GOOD_PLAN)]
    # Enough budget to make a model call, not enough for a search and the call that would use
    # its result.
    outcome, _, stub = await run_plan(script, retrieve=stub, deadline=time.monotonic() + 3.0)

    expect(failures, isinstance(outcome, PlanTurn), "the turn should still produce a plan")
    expect(failures, not stub.queries, "no search should start with the deadline that close")

    return failures


@check("a slow retrieval is reported as unavailable, not as empty")
async def _retrieval_timeout() -> list[str]:
    failures: list[str] = []
    original = agent_tools.RETRIEVAL_TIMEOUT_SECONDS
    agent_tools.RETRIEVAL_TIMEOUT_SECONDS = 0.05
    stub = RetrievalStub(chunks=[CHUNK], delay=0.3)

    try:
        script = [searches("fees"), answer(GOOD_PLAN)]
        outcome, client, _ = await run_plan(script, retrieve=stub)
    finally:
        agent_tools.RETRIEVAL_TIMEOUT_SECONDS = original

    if not isinstance(outcome, PlanTurn):
        return [f"a slow search must not fail the turn, got {outcome.code}"]

    payloads = [
        json.loads(message["content"])
        for message in client.calls[-1].messages
        if message.get("role") == "tool"
    ]

    expect(failures, len(payloads) == 1, "the tool call should still be answered")
    expect(failures, payloads[0]["available"] is False, "a timeout means the lookup did not run")
    expect(failures, payloads[0]["found"] is False, "and nothing was found")
    expect(
        failures,
        outcome.telemetry.retrieval_unavailable == 1,
        "telemetry should record the unavailable lookup",
    )
    expect(failures, not outcome.chunks, "no chunk should be attributed to a failed search")

    return failures


@check("retrieval that is down degrades the turn rather than failing it")
async def _retrieval_down() -> list[str]:
    failures: list[str] = []

    for label, stub in (
        ("unavailable", RetrievalStub(available=False)),
        ("raising", RetrievalStub(error=RuntimeError("pool exhausted"))),
    ):
        script = [searches("fees"), answer(plan_payload(reply=PLAIN_REPLY))]
        outcome, _, _ = await run_plan(script, retrieve=stub)

        expect(failures, isinstance(outcome, PlanTurn), f"{label} retrieval should not fail a turn")

        if isinstance(outcome, PlanTurn):
            expect(
                failures,
                outcome.telemetry.retrieval_unavailable == 1,
                f"{label}: telemetry should record it",
            )

    return failures


@check("malformed tool arguments are answered, never raised")
async def _bad_tool_args() -> list[str]:
    failures: list[str] = []
    stub = RetrievalStub(chunks=[CHUNK])
    script = [
        searches(
            ("search_government_information", "{not json"),
            ("search_government_information", {"query": "   "}),
            ("search_government_information", {"query": "fees", "limit": "lots"}),
            ("search_government_information", {"query": "fees", "agency": 7}),
            ("delete_everything", {}),
        ),
        answer(GOOD_PLAN),
    ]
    outcome, client, stub = await run_plan(script, retrieve=stub)

    if not isinstance(outcome, PlanTurn):
        return [f"bad arguments must not fail a turn, got {outcome.code}"]

    expect(failures, not stub.queries, "nothing should have been searched")
    expect(
        failures,
        outcome.telemetry.tool_errors == 5,
        f"expected five tool errors, got {outcome.telemetry.tool_errors}",
    )

    tool_messages = [
        message
        for message in client.calls[1].messages
        if message.get("role") == "tool"
    ]
    expect(failures, len(tool_messages) == 5, "every call gets a reply, or the next call is invalid")
    expect(
        failures,
        all(json.loads(message["content"])["available"] is False for message in tool_messages),
        "a refused call must not read as an empty corpus",
    )

    return failures


# --- The guards in front of a turn ---------------------------------------------------------------


@check("a second turn for one session is refused")
async def _session_busy() -> list[str]:
    failures: list[str] = []
    reset_locks()
    reset_quota()
    reset_breaker()

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowClient:
        model = "fake-model"

        async def complete(self, **_: Any) -> Any:
            started.set()
            await release.wait()

            return answer(GOOD_PLAN)

    async def first() -> Any:
        return await run_plan_turn(
            context=CONTEXT,
            fields=FIELDS,
            profile=PROFILE,
            retrieve=RetrievalStub(),
            client=SlowClient(),
            session_id="busy",
            user_id="u1",
        )

    task = asyncio.create_task(first())
    await started.wait()

    second = await run_plan_turn(
        context=CONTEXT,
        fields=FIELDS,
        profile=PROFILE,
        retrieve=RetrievalStub(),
        client=FakeModelClient([answer(GOOD_PLAN)]),
        session_id="busy",
        user_id="u1",
    )
    release.set()
    first_outcome = await task

    expect(
        failures,
        isinstance(second, TurnFailure) and second.code == "SESSION_BUSY",
        "the second turn should be refused, not queued",
    )
    expect(failures, isinstance(first_outcome, PlanTurn), "the first turn should still finish")

    # And the slot is released afterwards.
    again, _, _ = await run_plan([answer(GOOD_PLAN)], session_id="busy")
    expect(failures, isinstance(again, PlanTurn), "the lock must be released when a turn ends")

    return failures


@check("a user over the daily quota is refused before any call")
async def _quota() -> list[str]:
    failures: list[str] = []
    reset_locks()
    reset_quota()
    reset_breaker()

    record_usage("spender", settings.daily_token_quota + 1)
    client = FakeModelClient([answer(GOOD_PLAN)])

    outcome = await run_plan_turn(
        context=CONTEXT,
        fields=FIELDS,
        profile=PROFILE,
        retrieve=RetrievalStub(),
        client=client,
        session_id="s-quota",
        user_id="spender",
    )

    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "QUOTA_EXCEEDED",
        "a user over quota should be refused",
    )
    expect(failures, client.call_count == 0, "a refusal must cost nothing")

    reset_quota()

    return failures


@check("the circuit breaker fails fast after repeated transport failures")
async def _breaker() -> list[str]:
    failures: list[str] = []
    boom = ModelTransportError("down")

    # Three attempts in one turn is the threshold.
    await run_plan([boom, boom, boom])

    reset_locks()
    reset_quota()
    # Deliberately *not* resetting the breaker: this is the next caller.
    client = FakeModelClient([answer(GOOD_PLAN)])
    outcome = await run_plan_turn(
        context=CONTEXT,
        fields=FIELDS,
        profile=PROFILE,
        retrieve=RetrievalStub(),
        client=client,
        session_id="s-breaker",
        user_id="u1",
    )

    expect(
        failures,
        isinstance(outcome, TurnFailure) and outcome.code == "MODEL_UNAVAILABLE",
        "with the breaker open the next caller should fail fast",
    )
    expect(failures, client.call_count == 0, "the breaker must stop the call being made")

    reset_breaker()

    return failures


# --- The explain turn ------------------------------------------------------------------------


@check("the explain turn shares the machinery against its own schema")
async def _explain() -> list[str]:
    failures: list[str] = []
    reset_locks()
    reset_quota()
    reset_breaker()

    stub = RetrievalStub(chunks=[CHUNK])
    client = FakeModelClient(
        [
            searches("what is the fee"),
            answer(
                {
                    "reply": FEE_REPLY,
                    "citations": [{"chunk_id": 7, "source_url": "https://example.gov.ng/fees"}],
                },
            ),
        ],
    )

    outcome = await run_explain_turn(
        context="Explain the Email Address field.",
        retrieve=stub,
        client=client,
        session_id="s-explain",
        user_id="u1",
    )

    expect(failures, isinstance(outcome, ExplainTurn), "an explain turn should return a reply")

    if not isinstance(outcome, ExplainTurn):
        return failures

    expect(failures, outcome.grounding == "grounded", f"expected grounded, got {outcome.grounding}")
    expect(failures, len(outcome.response.citations) == 1, "the citation should verify")
    expect(failures, outcome.telemetry.kind == "explain", "telemetry should say which turn it was")
    expect(
        failures,
        client.calls[0].schema_name == "explain_response",
        "the explain schema should be the one enforced",
    )

    # And an unsupported claim is replaced, exactly as in a plan turn.
    reset_locks()
    ungrounded = FakeModelClient([answer({"reply": FEE_REPLY, "citations": []})] * 2)
    second = await run_explain_turn(
        context="Explain the fee.",
        retrieve=RetrievalStub(),
        client=ungrounded,
        session_id="s-explain-2",
        user_id="u1",
    )

    expect(
        failures,
        isinstance(second, ExplainTurn) and second.response.reply == UNVERIFIED_REPLY,
        "a twice-ungrounded explain reply must be replaced",
    )
    expect(failures, ungrounded.call_count == 2, "one repair, not more")

    return failures


# --- The contracts --------------------------------------------------------------------------


@check("every failure code has a sentence a person can read")
async def _catalogue() -> list[str]:
    failures: list[str] = []
    codes: tuple[FailureCode, ...] = (
        "MODEL_UNAVAILABLE",
        "MODEL_TIMEOUT",
        "BUDGET_EXCEEDED",
        "SCHEMA_FAILED",
        "SESSION_BUSY",
        "QUOTA_EXCEEDED",
    )

    expect(failures, set(FAILURE_MESSAGES) == set(codes), "the catalogue and the codes must agree")

    for code, message in FAILURE_MESSAGES.items():
        expect(failures, bool(message.strip()), f"{code} has no message")
        expect(failures, message[0].isupper(), f"{code}'s message should read as a sentence")
        expect(failures, message.rstrip().endswith("."), f"{code}'s message should end in a stop")
        expect(failures, code not in message, f"{code}'s message should not contain the code")
        expect(
            failures,
            not any(word in message.lower() for word in ("error", "exception", "schema", "token")),
            f"{code}'s message reads like a stack trace, not a sentence",
        )

    return failures


@check("telemetry carries no content, and neither does the log line")
async def _no_content() -> list[str]:
    failures: list[str] = []
    planted = {
        "secret query": "how much is the NIN slip",
        "reply": "I filled in your email address for you.",
        "label": "Mother's Maiden Name",
        "value": "adaeze@example.com",
    }

    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    try:
        stub = RetrievalStub(chunks=[CHUNK])
        script = [
            searches(planted["secret query"]),
            answer(
                plan_payload(
                    reply=planted["reply"],
                    actions=[fill("f_email", "email")],
                    missing=[
                        {
                            "field_id": "f_name",
                            "label": planted["label"],
                            "question": "What is your name?",
                        },
                    ],
                ),
            ),
        ]
        outcome, _, _ = await run_plan(script, retrieve=stub)
    finally:
        root.removeHandler(handler)

    if not isinstance(outcome, PlanTurn):
        return [f"expected a plan, got {outcome.code}"]

    rendered = repr(outcome.telemetry)
    log_text = "\n".join(records)

    for name, value in planted.items():
        expect(failures, value not in rendered, f"the telemetry record carries the {name}")
        expect(failures, value not in log_text, f"a log line carries the {name}")

    expect(failures, CHUNK.content not in log_text, "a log line carries retrieved content")
    expect(failures, "turn kind=plan" in log_text, "the turn line should have been emitted")
    expect(
        failures,
        "cost_usd=" in log_text,
        "the cost estimate should be visible in the turn line",
    )

    return failures


async def main() -> int:
    failures: list[str] = []

    print(f"checks ({len(CHECKS)}):")

    for name, func in CHECKS:
        try:
            case_failures = await func()
        except Exception as exc:  # noqa: BLE001  # a check that raised is a check that failed
            case_failures = [f"{name}: raised {type(exc).__name__}: {exc}"]

        print(f"  {'FAIL' if case_failures else 'ok  '}  {name}")
        failures += [f"{name}: {line}" for line in case_failures]

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for line in failures:
            print(f"  - {line}")

        return 1

    print("\nAll agent checks passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
