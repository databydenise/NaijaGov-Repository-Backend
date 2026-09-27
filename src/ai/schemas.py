"""
The one shape a model response may take.

`PlanResponse` is the source of truth. `PLAN_RESPONSE_SCHEMA` — the JSON schema handed to the
provider with `strict: true` — is *generated from it* at import (`build_strict_schema`), so the
two cannot drift: change the model and the provider schema changes with it. This is P4's job to
send; here it is only defined and proven consistent.

Two things the provider's strict mode cannot express, enforced by Pydantic instead:

- **Sizes.** `reply` ≤ 600 chars, `actions` ≤ 30, and so on. Strict structured outputs ignore
  length/count keywords, so `build_strict_schema` strips them and Pydantic is the real check:
  P4 parses the model's output through `PlanResponse`, and an over-limit payload fails and
  triggers the one repair pass.
- **Conditional requirements.** "`field_id` for every action but `pause`", "`value_ref` for
  `fill`/`select`" — there is no clean `if/then` in the strict subset, so these are model
  validators on `PlannedAction`.

The invariant that matters most: the model works with *keys*, never values. `value_ref` names a
`profile.*` or `chat.*` key; `extracted_data` is restricted to the profile field names, so a
model returning `nin` or `bvn` fails validation — those are not fields this product handles.
"""

from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from src.ai.constants import (
    MAX_ACTIONS,
    MAX_CITATIONS,
    MAX_EXAMPLE_CHARS,
    MAX_EXPLAIN_CITATIONS,
    MAX_EXPLANATION_CHARS,
    MAX_MISSING,
    MAX_REPLY_CHARS,
)
from src.profiles.constants import PROFILE_FIELDS

# The action types the model may return. This list now matches `ACTION_TYPES` in
# `sessions/constants.py`, which mirrors `naijagov-extension/src/shared/actions.ts`: P3 added
# `clickSafe` so the two agree and the guard's BLOCKED_BUTTON check has something to reject.
# Still to be confirmed against `actions.ts` by eye before P4 ships — widening or renaming this
# list is a two-repo change, not a one-line edit here.
ActionType = Literal[
    "fill",
    "select",
    "check",
    "highlight",
    "scroll",
    "explain",
    "clickSafe",
    "pause",
]

# Actions that must name a field. `pause` is the only one that need not — it hands control back
# to the human and may reference no single control.
_ACTIONS_REQUIRING_FIELD: Final = frozenset(
    {"fill", "select", "check", "highlight", "scroll", "explain", "clickSafe"},
)
# Actions that must carry a value reference — the two that write into the page.
_ACTIONS_REQUIRING_VALUE_REF: Final = frozenset({"fill", "select"})


class StrictModel(BaseModel):
    """
    Base for every response model.

    `extra="forbid"` is both a Pydantic guard (an unexpected key fails) and what makes the
    generated schema carry `additionalProperties: false` on its own, before `build_strict_schema`
    reasserts it. A model that returns a field we did not define is a bug, not noise.
    """

    model_config = ConfigDict(extra="forbid")


class ValueRef(StrictModel):
    """
    A reference to a value the backend holds, never the value itself.

    `source="profile"` points at a stored profile key; `source="chat"` at something the user
    typed in this conversation and the session kept. The guard resolves it to a real value after
    validation, so the model can never write a phone number it was never shown.
    """

    source: Literal["profile", "chat"]
    key: str


class PlannedAction(StrictModel):
    """
    One thing to do to the page.

    `field_id` references an id from the snapshot — never a selector, never one the model
    invented; the guard drops an id that was not in the snapshot. Every field but `pause` names
    one, and `fill`/`select` also carry a `value_ref` rather than a literal value.
    """

    type: ActionType
    field_id: str | None = None
    value_ref: ValueRef | None = None
    checked: bool | None = None
    reason: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check_conditional_fields(self) -> "PlannedAction":
        if self.type in _ACTIONS_REQUIRING_FIELD and not self.field_id:
            raise ValueError(f"action '{self.type}' requires a field_id")
        if self.type in _ACTIONS_REQUIRING_VALUE_REF and self.value_ref is None:
            raise ValueError(f"action '{self.type}' requires a value_ref")

        return self


class Citation(StrictModel):
    """
    Provenance for a factual claim. `chunk_id` ties the claim to a retrieved chunk; the guard
    (P3) drops a citation whose id was not among the ids actually retrieved.
    """

    chunk_id: int
    source_url: str


class MissingItem(StrictModel):
    """
    A field the plan needs a value for and does not have.

    Missing data is a structured output, not a sentence buried in `reply`, so the panel can
    render `question` as a prompt the user answers rather than prose they have to parse.
    """

    field_id: str
    label: str
    question: str


def _build_extracted_data_model() -> type[BaseModel]:
    """
    A submodel with one nullable field per profile key, built from `PROFILE_FIELDS`.

    The spec types `extracted_data` as `dict[str, str]`, but strict structured outputs cannot
    express an open string-keyed object — every property must be named. Naming them from
    `PROFILE_FIELDS` is also exactly the allowlist the spec calls for: `nin`, `bvn` and anything
    else are rejected as unexpected keys (`extra="forbid"`), and the allowlist can never drift
    from the profile because it *is* the profile's field list. Unknown values come back null and
    the guard ignores them.
    """
    fields: dict[str, Any] = {name: (str | None, None) for name in PROFILE_FIELDS}

    return create_model(
        "ExtractedData",
        __config__=ConfigDict(extra="forbid"),
        __doc__=(
            "Values the user typed in chat, keyed by profile field name. One nullable field "
            "per profile key; unknown keys fail validation."
        ),
        **fields,
    )


ExtractedData = _build_extracted_data_model()


class PlanResponse(StrictModel):
    """
    The whole of what `/plan` may return. P4 parses the model's output through this; anything
    that fails is repaired once, then dropped.
    """

    reply: str = Field(max_length=MAX_REPLY_CHARS)
    actions: list[PlannedAction] = Field(default_factory=list, max_length=MAX_ACTIONS)
    extracted_data: ExtractedData
    citations: list[Citation] = Field(default_factory=list, max_length=MAX_CITATIONS)
    missing: list[MissingItem] = Field(default_factory=list, max_length=MAX_MISSING)


class ExplainResponse(StrictModel):
    """
    The whole of what `/explain` may return: a sentence or two about one field, and its sources.

    Deliberately minimal. `/explain` describes a field — it never writes to the page, so there is
    no action list and no `value_ref`, and the smaller schema is most of why the explain turn is
    cheaper than a plan turn. "No actions" is a property of this class rather than a rule someone
    has to remember: there is no field an action could arrive in.

    Its three caps are its own, not `PlanResponse`'s. An explanation is two or three sentences in
    a card beside one field (`MAX_EXPLANATION_CHARS`), an example is a value rather than a
    sentence (`MAX_EXAMPLE_CHARS`), and three sources is as many as a side panel can show without
    the user reading past them (`MAX_EXPLAIN_CITATIONS`). Sharing the plan's numbers, as this
    model did before P6, meant a 600-character explanation and five sources were both valid —
    neither of which the panel has room for.

    `example` is nullable because most labels make one meaningless: "Declaration" has no example
    value, and inventing one is the failure this project spends most of its code preventing. A
    model that has nothing to show returns `null`, which strict mode requires it to send.

    Grounding is checked the same way as a plan's, against the same `Citation` — a fee or a format
    requirement with no retrieved chunk behind it is unverified whichever endpoint said it.
    """

    explanation: str = Field(max_length=MAX_EXPLANATION_CHARS)
    example: Annotated[str, Field(max_length=MAX_EXAMPLE_CHARS)] | None = None
    citations: list[Citation] = Field(default_factory=list, max_length=MAX_EXPLAIN_CITATIONS)


# Keywords the provider's strict mode does not support. Kept on the Pydantic model (where they
# are enforced) and stripped from the schema the provider sees, so a strict call is never
# rejected for carrying one.
_UNSUPPORTED_SCHEMA_KEYS: Final = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "default",
    },
)


def _make_strict(node: Any) -> None:  # noqa: ANN401  # walks arbitrary JSON-schema nodes
    """
    Rewrite a JSON-schema node in place for provider strict mode.

    On every object: mark every property required (strict mode has no optional properties —
    a field is made optional by being nullable, which Pydantic already renders) and forbid
    extra properties. Everywhere: drop the length/format keywords strict mode rejects.
    """
    if isinstance(node, dict):
        for key in _UNSUPPORTED_SCHEMA_KEYS & node.keys():
            del node[key]

        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False

        for value in node.values():
            _make_strict(value)
        return

    if isinstance(node, list):
        for item in node:
            _make_strict(item)


def build_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema, rewritten for OpenAI strict structured outputs."""
    schema = model.model_json_schema()
    _make_strict(schema)

    return schema


# Handed to the provider as the `json_schema` (with `strict: true`) by P4. Generated here so it
# can never disagree with `PlanResponse`.
PLAN_RESPONSE_SCHEMA: Final[dict[str, Any]] = build_strict_schema(PlanResponse)


# Handed to the provider on an `/explain` call. Generated the same way, from the same base, so
# the two response contracts cannot drift apart in how strictly they are enforced.
EXPLAIN_RESPONSE_SCHEMA: Final[dict[str, Any]] = build_strict_schema(ExplainResponse)
