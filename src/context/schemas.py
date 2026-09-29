"""
`/context` request and response models.

The request describes a page's *structure*: which fields exist, what they are labelled,
what type they are. There is no field for what a user typed into any of them, and
`extra="forbid"` means a client that starts sending one gets a 400 rather than having it
quietly ignored. A permissive schema here is how page values end up in a database.
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.constants import SAFE_MESSAGE_MARKER
from src.context.constants import (
    FORBIDDEN_FIELD_KEYS,
    MAX_BUTTONS,
    MAX_FIELD_ID_LENGTH,
    MAX_FIELDS,
    MAX_HEADING_LENGTH,
    MAX_HEADINGS,
    MAX_LABEL_LENGTH,
    MAX_LINKS,
    MAX_OPTIONS,
    MAX_PAGE_HASH_LENGTH,
    MAX_SENSITIVE_FLAGS,
    MAX_TITLE_LENGTH,
    MAX_URL_LENGTH,
)

Label = Annotated[str, Field(max_length=MAX_LABEL_LENGTH)]
FieldId = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_ID_LENGTH)]


class SnapshotModel(BaseModel):
    """Base for everything in a snapshot: unknown keys are an error, not noise."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PageField(SnapshotModel):
    """
    One form control, as the content script found it.

    `sensitive` is the content script's judgement and is passed through, never re-decided
    here: the DOM is where that question can be answered.
    """

    field_id: FieldId
    label: Label = ""
    type: Label = ""
    options: list[Label] = Field(default_factory=list, max_length=MAX_OPTIONS)
    required: bool = False
    sensitive: bool = False

    @model_validator(mode="before")
    @classmethod
    def _no_value_keys(cls, data: object) -> object:
        """
        Refuse a field object carrying what the user typed.

        `extra="forbid"` would reject these anyway, but with Pydantic's generic wording and
        without saying why it matters. This is a tripwire for the rule that page values
        never reach this service, so it says so in its own words — and names only the key,
        never what was in it.
        """
        if not isinstance(data, dict):
            return data

        offending = sorted(FORBIDDEN_FIELD_KEYS.intersection(map(str, data)))

        if offending:
            # Marked safe so the handler shows it verbatim. Everything interpolated here
            # is a key name from our own constant, never a submitted value.
            message = (
                f"{SAFE_MESSAGE_MARKER} A field may not carry "
                f"{', '.join(offending)}. This service records what a page asks for, "
                f"never what anyone typed into it."
            )
            raise ValueError(message)

        return data


class PageButton(SnapshotModel):
    """A button. `text` is its visible label, which is how a submit is recognised."""

    field_id: FieldId
    text: Label = ""
    sensitive: bool = False


class PageLink(SnapshotModel):
    """
    A navigation link: an `<a href>` that leads somewhere else.

    Separate from `PageButton` because the two are answered differently. A button may be pressed
    on the user's behalf when the extension has cleared it; a link never is — following one
    navigates the tab away, and that stays the user's decision. The guard refuses a `clickSafe`
    on anything in this list, and `ai/context.py` renders it under a heading that says so.

    `href` is origin and path only: the extension strips the query string before sending, for the
    same reason it strips it from `url`. A `method="get"` form puts what the user typed into the
    next page's URL, and a query string is therefore page *values* wearing a link's clothes.

    `sensitive` is the content script's judgement, passed through unchanged. It marks a link the
    model should not steer the user towards — a payment page, say — and deliberately does **not**
    raise a checkpoint: a masthead carries "Make a payment" on every page of a portal, including
    the pages that have no payment on them.
    """

    field_id: FieldId
    text: Label = ""
    href: Annotated[str, Field(max_length=MAX_URL_LENGTH)] = ""
    external: bool = False
    sensitive: bool = False


class SensitiveFlag(SnapshotModel):
    """
    A field the content script marked as a checkpoint.

    `reason` is a short category — "password", "otp" — not the content of anything.
    """

    field_id: FieldId
    reason: Label = ""


class ContextRequest(SnapshotModel):
    """A page snapshot: structure and labels, never values."""

    tab_id: int
    url: Annotated[str, Field(min_length=1, max_length=MAX_URL_LENGTH)]
    title: Annotated[str, Field(max_length=MAX_TITLE_LENGTH)] = ""

    # The client's hash. Compared against the server's and otherwise unused: trusting it
    # would let a stale plan be replayed against a page that has since changed.
    page_hash: Annotated[str, Field(max_length=MAX_PAGE_HASH_LENGTH)] = ""

    headings: list[Annotated[str, Field(max_length=MAX_HEADING_LENGTH)]] = Field(
        default_factory=list,
        max_length=MAX_HEADINGS,
    )
    fields: list[PageField] = Field(default_factory=list, max_length=MAX_FIELDS)
    buttons: list[PageButton] = Field(default_factory=list, max_length=MAX_BUTTONS)
    links: list[PageLink] = Field(default_factory=list, max_length=MAX_LINKS)
    sensitive_flags: list[SensitiveFlag] = Field(
        default_factory=list,
        max_length=MAX_SENSITIVE_FLAGS,
    )

    # The content script's own doubt about what it read. Recorded, not acted on here.
    ambiguous: bool = False


class WorkflowOut(BaseModel):
    """The matched workflow, as the panel's header shows it."""

    id: str
    name: str
    agency: str


class StepOut(BaseModel):
    """Where in the workflow this page is. `total` is what makes "Step 1 of 2" sayable."""

    id: str
    name: str
    index: int
    total: int
    is_final: bool


class Coverage(BaseModel):
    """How much of the step was found. `missing` holds stored labels, for a person."""

    matched: int
    missing: list[str]


class FieldHint(BaseModel):
    """
    A field on the page that has a rule.

    `is_placeholder` rides along on every hint so the panel can label demo content. Without
    it, an invented requirement shown beside a `.gov.ng` source is indistinguishable from
    an official one.
    """

    field_id: str
    rule_id: str
    is_placeholder: bool


class Checkpoint(BaseModel):
    """That a checkpoint exists and how many fields it covers. Never which, or why."""

    present: bool
    count: int


class ContextResponse(BaseModel):
    """
    What the panel needs to leave READING.

    On an unsupported page everything after `message` is absent: there is no workflow to
    describe, and `supported: false` is a normal outcome rather than an error.
    """

    session_id: uuid.UUID
    supported: bool

    # The server's hash, always. The extension adopts it so `/plan`'s PAGE_CHANGED check
    # compares like with like.
    page_hash: str
    cached: bool = False

    message: str | None = None
    workflow: WorkflowOut | None = None
    step: StepOut | None = None
    confidence: Literal["high", "low"] | None = None
    coverage: Coverage | None = None
    field_hints: list[FieldHint] = Field(default_factory=list)
    checkpoint: Checkpoint | None = None
    rules_loaded: int = 0
