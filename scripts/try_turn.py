"""
Run one real turn against the live model and the live corpus, by hand.

    python -m scripts.try_turn                      # all three scenarios
    python -m scripts.try_turn --scenario grounded  # just one
    python -m scripts.try_turn --message "…"        # your own question
    python -m scripts.try_turn --explain "Phone Number"
    python -m scripts.try_turn --scenario fill --repeat 5   # median duration

This is the opposite of `scripts/check_agent.py`, and the two are both needed. That one proves
the machinery — budgets, caps, repairs, failure codes — against a scripted fake, offline and for
free. This one proves the things a fake cannot:

- that OpenAI accepts `PLAN_RESPONSE_SCHEMA` with `strict: true` (unverified since P2; a
  rejection is a `ModelPermanentError`, not something a retry fixes);
- that the model still calls the search tool now that the response schema is attached to every
  call rather than only the last one (`src/agent/turn.py`);
- that the prompts produce sane actions against a real snapshot.

**It spends money and makes network calls.** Every scenario is one to three model calls plus an
embedding call per search. It needs `OPENAI_API_KEY` and a reachable `DATABASE_URL`, and it
refuses to run without the key rather than reporting a misleading failure.

Nothing here is written to the database. The snapshot below is a hand-written stand-in for what
the extension sends, using the field labels from `src/database/seed/data/steps.json` so the
labels match the seeded registry.
"""

import argparse
import asyncio
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from src.agent.explain import run_explain_turn
from src.agent.plan import run_plan_turn
from src.agent.schemas import ExplainTurn, PlanTurn, TurnFailure, TurnTelemetry
from src.ai import client as model_client
from src.ai.context import StepView, render_page_block, render_plan_context
from src.ai.prompt_loader import PROMPT_VERSION
from src.config import settings
from src.context.schemas import PageButton, PageField
from src.documents.service import search_government_information
from src.logging import RedactingFilter, configure_logging

# --- The snapshot, as the extension would send it ---------------------------------------------
#
# The `demo_reg` applicant step. Labels are copied from the seed so a mismatch here is a real
# mismatch, not an artefact of this file.

STEP = StepView(name="Applicant Information", index=1, total=2)

FIELDS = [
    PageField(field_id="f1", label="Full Name", type="text", required=True),
    PageField(field_id="f2", label="Email Address", type="email", required=True),
    PageField(field_id="f3", label="Phone Number", type="tel", required=True),
    PageField(
        field_id="f4",
        label="State",
        type="select",
        required=True,
        options=["Abia", "Kano", "Lagos", "Rivers"],
    ),
    PageField(field_id="f5", label="Local Government Area", type="text", required=True),
    PageField(
        field_id="f6",
        label="Business Type",
        type="select",
        required=True,
        options=["Sole Proprietor", "Limited Liability", "Partnership"],
    ),
    PageField(field_id="f7", label="Password", type="password", required=True, sensitive=True),
]
BUTTONS = [
    PageButton(field_id="b1", text="Continue"),
    PageButton(field_id="b2", text="Submit Application", sensitive=True),
]
# What the content script's `sensitive_flags` would carry. Honoured as well as the field's own
# flag, not instead of it — either one blocks.
BLOCKED = ("f7",)

# The demo account's fictional profile (`src/demo.py`). `lga` is deliberately absent, so a turn
# that needs it has to ask rather than invent one.
PROFILE: dict[str, str | None] = {
    "full_name": "Adaeze Okonkwo",
    "email": "demo@example.com",
    "phone": "08000000000",
    "address": "12 Marina Road, Lagos Island",
    "state": "Lagos",
    "lga": None,
}
CHAT_VALUES = {"lga": "Ikeja"}


@dataclass(frozen=True)
class Scenario:
    """One question, and what a correct answer to it looks like."""

    name: str
    message: str
    expectation: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="fill",
        message="Please fill in whatever you can on this page for me.",
        expectation=(
            "Fills from the profile and from chat, refuses the Password field, and asks for "
            "Business Type rather than guessing it. Probably no search."
        ),
    ),
    Scenario(
        name="grounded",
        message="How much does it cost to renew a driver's licence, and how long does it take?",
        expectation=(
            "Searches, then answers only what the FRSC corpus says, with a citation. This is "
            "the scenario that proves tool calling still fires with the schema attached."
        ),
    ),
    Scenario(
        name="unanswerable",
        message="What is the penalty for filing my annual returns late in Kano State?",
        expectation=(
            "Searches, finds nothing (the corpus is FRSC driver's licence material), and says "
            "it has no official guidance. An invented fee or deadline here is a failure — this "
            "is the prototype regression the whole grounding layer exists to prevent."
        ),
    ),
)


class TelemetryCapture(logging.Handler):
    """Keeps the runner's own turn lines, so they can be pasted into a commit message."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        # The same redaction the real handlers apply, so what is printed is what would be logged.
        self.addFilter(RedactingFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def print_plan(outcome: PlanTurn) -> None:
    """The guarded plan, as the side panel would show it."""
    plan = outcome.plan

    print(f"  reply       : {plan.reply}")
    print(f"  grounding   : {outcome.grounding}")

    if plan.reply_replaced:
        print("  NOTE        : the reply was replaced — a claim survived two ungrounded attempts")

    print(f"  approved    : {len(plan.approved)}")
    for action in plan.approved:
        detail = f"{action.type} {action.field_id or ''}".strip()
        if action.value is not None:
            detail += f"  ←  {action.source} = {action.value!r}"
        if action.reason:
            detail += f"  ({action.reason})"
        if action.suspicious:
            detail += "  [SUSPICIOUS: key and label disagree]"
        print(f"      {detail}")

    print(f"  rejected    : {len(plan.rejected)}")
    for rejected in plan.rejected:
        print(f"      {rejected.code} {rejected.field_id or ''}  — {rejected.message}")

    print(f"  missing     : {len(plan.missing)}")
    for item in plan.missing:
        print(f"      {item.field_id} {item.label}  — {item.question}")

    print(f"  citations   : {len(plan.citations)}")
    for citation in plan.citations:
        print(f"      chunk {citation.chunk_id}  {citation.source_url}")


def print_explain(outcome: ExplainTurn) -> None:
    print(f"  reply       : {outcome.response.reply}")
    print(f"  grounding   : {outcome.grounding}")
    for citation in outcome.response.citations:
        print(f"      chunk {citation.chunk_id}  {citation.source_url}")


def print_telemetry(telemetry: TurnTelemetry) -> None:
    """The numbers that matter for the 6-second target and the cost estimate."""
    print(f"  duration    : {telemetry.duration_ms:.0f} ms   phases={telemetry.phase_ms}")
    print(
        f"  tokens      : prompt={telemetry.prompt_tokens} "
        f"completion={telemetry.completion_tokens} "
        f"cost_usd={telemetry.estimated_cost_usd:.6f}",
    )
    print(
        f"  calls       : model={telemetry.model_calls} retries={telemetry.retries} "
        f"tools={telemetry.tool_calls} tool_errors={telemetry.tool_errors} "
        f"retrieval_unavailable={telemetry.retrieval_unavailable} "
        f"chunks={telemetry.chunks_retrieved}",
    )
    print(f"  repair      : {telemetry.repair_ran} ({telemetry.repair_reason or 'none'})")

    if telemetry.estimated_cost_usd == 0.0 and telemetry.total_tokens > 0:
        print("  NOTE        : cost reads 0.0 because MODEL_RATES is not filled in yet")

    if telemetry.retrieval_unavailable:
        print(
            "  WARNING     : a lookup did not run. An empty answer here is not evidence the "
            "corpus lacks the material — check DATABASE_URL and the embedding key.",
        )


async def one_plan_turn(
    message: str,
    session_suffix: str,
    client: model_client.ModelClient,
) -> PlanTurn | TurnFailure:
    """One plan turn, through the real renderer, the real corpus and the real model."""
    context = render_plan_context(
        step=STEP,
        fields=FIELDS,
        buttons=BUTTONS,
        profile=PROFILE,
        chat_values=CHAT_VALUES,
        history=[],
        message=message,
    )

    return await run_plan_turn(
        context=context,
        fields=FIELDS,
        buttons=BUTTONS,
        profile=PROFILE,
        chat_values=CHAT_VALUES,
        blocked_field_ids=BLOCKED,
        retrieve=search_government_information,
        client=client,
        session_id=f"try-turn-{session_suffix}",
        user_id="try-turn",
    )


async def one_explain_turn(
    label: str,
    client: model_client.ModelClient,
) -> ExplainTurn | TurnFailure:
    """
    One explain turn.

    P6 owns the real explain renderer; until it exists the context is the page block plus the
    question, which is enough to exercise the entry point end to end.
    """
    context = (
        f"{render_page_block(STEP, FIELDS, BUTTONS)}\n\n"
        f"Explain the field labelled \"{label}\" to the user."
    )

    return await run_explain_turn(
        context=context,
        retrieve=search_government_information,
        client=client,
        session_id="try-turn-explain",
        user_id="try-turn",
    )


async def run(args: argparse.Namespace) -> int:
    if not model_client.is_configured():
        print(
            "OPENAI_API_KEY is not set. This script makes real calls; there is nothing to "
            "fall back to.\nRun scripts.check_agent instead for the offline checks.",
        )

        return 2

    capture = TelemetryCapture()
    logging.getLogger("src.agent.telemetry").addHandler(capture)

    # One client for the process, as in production: each one builds its own connection pool.
    client = model_client.OpenAIClient()

    print(f"model={settings.model_name}  prompt_version={PROMPT_VERSION}")
    print("This makes real, paid calls.\n")

    durations: list[float] = []
    failed = False

    if args.explain:
        print(f"--- explain: {args.explain} ---")
        outcome = await one_explain_turn(args.explain, client)

        if isinstance(outcome, TurnFailure):
            print(f"  FAILED      : {outcome.code}\n  message     : {outcome.message}")
            failed = True
        else:
            print_explain(outcome)
            print_telemetry(outcome.telemetry)
            durations.append(outcome.telemetry.duration_ms)
    else:
        chosen = _chosen_scenarios(args)

        for index, scenario in enumerate(chosen):
            print(f"--- {scenario.name} (run {index + 1} of {len(chosen)}) ---")
            print(f"  asking      : {scenario.message}")
            print(f"  expecting   : {scenario.expectation}")

            outcome = await one_plan_turn(
                scenario.message,
                f"{scenario.name}-{index}",
                client,
            )

            if isinstance(outcome, TurnFailure):
                print(f"  FAILED      : {outcome.code}\n  message     : {outcome.message}")
                print_telemetry(outcome.telemetry)
                failed = True
            else:
                print_plan(outcome)
                print_telemetry(outcome.telemetry)
                durations.append(outcome.telemetry.duration_ms)

            print()

    if durations:
        print(f"durations (ms): {[round(value) for value in durations]}")
        print(f"median        : {statistics.median(durations):.0f} ms  (target: under 6000)")

    print("\n--- telemetry lines, for the commit message ---")
    for line in capture.lines:
        print(line)

    return 1 if failed else 0


def _chosen_scenarios(args: argparse.Namespace) -> list[Scenario]:
    """The scenarios to run, repeated if a median is wanted."""
    if args.message:
        base: Sequence[Scenario] = (
            Scenario(name="custom", message=args.message, expectation="whatever you asked for"),
        )
    elif args.scenario == "all":
        base = SCENARIOS
    else:
        base = [scenario for scenario in SCENARIOS if scenario.name == args.scenario]

    return list(base) * args.repeat


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real agent turn against the live model.")
    parser.add_argument(
        "--scenario",
        choices=("all", *(scenario.name for scenario in SCENARIOS)),
        default="all",
        help="which prepared question to ask (default: all three)",
    )
    parser.add_argument("--message", help="ask your own question instead of a prepared one")
    parser.add_argument("--explain", metavar="LABEL", help="run an explain turn for one field")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each scenario N times and report the median duration",
    )
    args = parser.parse_args()

    configure_logging()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
