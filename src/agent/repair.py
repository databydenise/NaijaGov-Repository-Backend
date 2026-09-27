"""
The repair pass: one diagnosis, one instruction, one more attempt.

A repair is not a retry. A retry re-sends the same request because the transport failed; a repair
re-asks because the *content* was wrong, and it says exactly what was wrong. Three diagnoses earn
one, and nothing else does:

- **the schema failed** — the answer would not parse, so the errors go back with it;
- **the guard returned `unverified`** — a factual claim with nothing retrieved behind it;
- **every action was rejected as malformed** — the model's action shapes were unusable.

A rejection for a legitimate reason — a sensitive field, an id that is not on the page — is not a
repair. The guard was right, and asking again invites the model to argue with it.

Two details in `repair_messages` are the whole reason this is its own module:

- The repair **reuses the original context** — the same system prompt and the same rendered page —
  so the second attempt is answering the same question, not a summary of its own failure.
- It **does not carry the failed answer back as an assistant turn.** A model shown its own wrong
  output tends to defend it; one shown only the instruction tends to follow it.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import ValidationError

from src.ai.client import Message, user_message

RepairReason = Literal["schema", "unverified", "malformed_actions"]

# How many validation errors are fed back. Past a handful the model is not reading them, and the
# instruction is competing with the page snapshot for its attention.
MAX_REPORTED_ERRORS: Final = 6

_SCHEMA_PREFIX: Final = (
    "Your previous answer did not match the required response format and was discarded. "
    "The problems were:"
)

_SCHEMA_SUFFIX: Final = (
    "Answer the same question again, as a single JSON object in exactly the required format. "
    "Nothing else is acceptable — not an explanation, not a partial object."
)

_UNVERIFIED_INSTRUCTION: Final = (
    "Your previous answer made a factual claim about the government process — a fee, a timeline, "
    "a document, an eligibility rule — that none of the retrieved sources supports, so it was "
    "discarded. Answer again. Either state only what a retrieved source says and cite its "
    "chunk_id, or say plainly that you do not have official guidance on this and suggest where "
    "the user might check. Do not answer from your own knowledge."
)

_MALFORMED_PREFIX: Final = (
    "Every action in your previous answer was unusable and was discarded. The reasons were:"
)

_MALFORMED_SUFFIX: Final = (
    "Answer again. Every field_id must be one listed in the page snapshot, exactly as written "
    "there. Every value must be a value_ref naming a profile or chat key — never a literal "
    "value. If a field has no value available, leave it out and put it in `missing` instead."
)


def schema_errors(exc: ValidationError) -> list[str]:
    """
    Validation failures as short lines the model can act on.

    `include_input=False` matters: pydantic's error objects carry the rejected value, and the
    rejected value can be something the model echoed out of the user's data. The location and the
    reason are enough to fix the shape, and nothing else is put in a string that gets sent
    anywhere.
    """
    lines: list[str] = []

    for error in exc.errors(include_url=False, include_input=False)[:MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        lines.append(f"- {location}: {error.get('msg', 'invalid')}")

    return lines


def error_signatures(exc: ValidationError) -> tuple[str, ...]:
    """
    The same failures as `schema_errors`, reduced to something safe to log.

    One entry per failure, as `location:type` — `reply:string_too_long`, `actions.0:value_error`.
    Both halves are ours: the location is a path through our own model, and the type is pydantic's
    fixed slug. Neither can carry a value the model wrote, which is why this may go in the
    telemetry while `schema_errors` may not.

    It earns its place because a repair pass that fires on most turns doubles the latency and the
    cost of those turns, and "errors=1" in a log line does not say which limit to loosen or which
    prompt line to sharpen. This does.
    """
    return tuple(
        f"{'.'.join(str(part) for part in error.get('loc', ())) or '(root)'}:"
        f"{error.get('type', 'unknown')}"
        for error in exc.errors(include_url=False, include_input=False)[:MAX_REPORTED_ERRORS]
    )


def repair_instruction(
    reason: RepairReason,
    *,
    errors: Sequence[str] = (),
    rejection_codes: Mapping[str, int] | None = None,
) -> str:
    """The instruction for one diagnosis. Our own words throughout; nothing echoed back."""
    if reason == "schema":
        body = "\n".join(errors) or "- the answer was not valid JSON for the required schema"

        return f"{_SCHEMA_PREFIX}\n{body}\n\n{_SCHEMA_SUFFIX}"

    if reason == "malformed_actions":
        codes = rejection_codes or {}
        body = "\n".join(f"- {code} ({count})" for code, count in sorted(codes.items()))

        return f"{_MALFORMED_PREFIX}\n{body or '- MALFORMED'}\n\n{_MALFORMED_SUFFIX}"

    return _UNVERIFIED_INSTRUCTION


def repair_messages(base: Sequence[Message], instruction: str) -> list[Message]:
    """
    The repair call's messages: the conversation as it stood, then the instruction.

    `base` is everything up to but *not including* the answer that failed — the system prompt, the
    rendered context, and any tool exchange from the first attempt. The tool results stay, and
    have to: an `unverified` repair asks the model to cite a retrieved chunk, which it cannot do
    if the retrieved chunks are no longer in front of it. What is dropped is the failed answer
    itself, so the model is not defending its own mistake.
    """
    return [*base, user_message(instruction)]
