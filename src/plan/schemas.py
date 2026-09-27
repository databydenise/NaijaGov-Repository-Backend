"""
`/plan` request and response models.

The request is `/context`'s snapshot plus a message, and it reuses `PageField`, `PageButton` and
`SensitiveFlag` rather than restating them — including the tripwire that refuses a field object
carrying a `value`. Two definitions of one payload shape is how one of them quietly loses a
check, and the check it would lose here is the one that keeps what a citizen typed into a
government form out of this service.

The response is the other direction, and the one place in this project where **real values
travel**: they are going back to the browser the user is sitting at, to fill their own form. Every
row carries where its value came from, because a fill the user cannot check is a fill they cannot
refuse.
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.context.constants import (
    MAX_BUTTONS,
    MAX_FIELDS,
    MAX_HEADING_LENGTH,
    MAX_HEADINGS,
    MAX_PAGE_HASH_LENGTH,
    MAX_SENSITIVE_FLAGS,
    MAX_URL_LENGTH,
)
from src.context.schemas import PageButton, PageField, SensitiveFlag, SnapshotModel
from src.plan.constants import MAX_CLIENT_PLAN_ID_CHARS, MAX_MESSAGE_CHARS


class PlanRequest(SnapshotModel):
    """
    What the panel sends to plan a turn: the session, the page as it is *now*, and the message.

    The snapshot is re-sent rather than read back from the session. It could have been stored at
    `/context`, but the guard has to check every action against what is on screen now, and a page
    that changed since is the case this endpoint exists to refuse. Re-sending costs a larger body;
    reusing would cost correctness.

    `url` is here for the hash, not for a lookup: the server recomputes `page_hash` from the URL
    path and the fields, and a client-supplied hash would let a stale plan be replayed against a
    page that has since changed.
    """

    session_id: uuid.UUID
    url: Annotated[str, Field(min_length=1, max_length=MAX_URL_LENGTH)]

    # The client's own hash. Compared, never trusted; the server's is authoritative.
    page_hash: Annotated[str, Field(max_length=MAX_PAGE_HASH_LENGTH)] = ""

    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]

    # The panel's correlation id, echoed back untouched so it can match a response to the
    # request it made. Never used to look anything up.
    client_plan_id: Annotated[str, Field(max_length=MAX_CLIENT_PLAN_ID_CHARS)] | None = None

    headings: list[Annotated[str, Field(max_length=MAX_HEADING_LENGTH)]] = Field(
        default_factory=list,
        max_length=MAX_HEADINGS,
    )
    fields: list[PageField] = Field(default_factory=list, max_length=MAX_FIELDS)
    buttons: list[PageButton] = Field(default_factory=list, max_length=MAX_BUTTONS)
    sensitive_flags: list[SensitiveFlag] = Field(
        default_factory=list,
        max_length=MAX_SENSITIVE_FLAGS,
    )


class PlannedActionOut(BaseModel):
    """
    One row of the fill preview: what would be written, where, and on whose authority.

    `source` is written for display — "Your profile", "You told me just now" — so the preview can
    show provenance on every row without the panel inventing wording for it. `source_ref` is the
    same fact in our own terms (`profile.email`), because the extension's `Action` carries a
    structured `ValueSource` and reconstructing one by reading an English sentence would be
    absurd. Both are present on every write; both are absent on an action that writes nothing.

    `suspicious` is the guard's one soft outcome: the value resolved and the field accepted it,
    but the key and the label disagree. The row is offered with a warning rather than dropped,
    because the alternative refuses legitimate fills.
    """

    action_id: str
    type: str
    field_id: str | None = None
    label: str | None = None
    value: str | None = None
    checked: bool | None = None
    source: str | None = None
    source_ref: str | None = None
    reason: str | None = None
    note: str | None = None
    suspicious: bool = False


class RejectedActionOut(BaseModel):
    """
    One thing the model asked for that the guard refused.

    Reported rather than silently dropped: "1 suggestion was blocked (password field)" reads as
    care, while silence reads as an unreliable tool and hides our own bugs. `reason` is the
    guard's sentence for the user; the code behind it goes to the log, not the panel.
    """

    field_id: str | None = None
    label: str | None = None
    reason: str


class MissingItemOut(BaseModel):
    """A field the plan has no value for, and the question that would get one."""

    field_id: str
    label: str
    question: str


class CitationOut(BaseModel):
    """
    A source behind a factual claim, as the panel shows it.

    `retrieved_at` is when the corpus last read that source, not when this answer was written. A
    citation whose date we do not hold shows none, rather than borrowing today's and implying the
    source was checked this morning.
    """

    title: str
    url: str
    retrieved_at: str | None = None


class StepOut(BaseModel):
    """Where this page sits in its workflow. Absent for a page the registry does not know."""

    id: str
    name: str
    index: int
    total: int


class PlanResponse(BaseModel):
    """
    What the panel needs to render a preview and let the user approve it.

    `plan_id` is what comes back on approval, so `/results` can be matched against what was
    actually proposed rather than against whatever the extension says it did.

    `grounding` of `unverified` means the panel shows the reply with a caveat; `not_required` is
    the common case and shows nothing; `grounded` means every claim traces to a cited source.
    """

    plan_id: str
    cached: bool = False
    client_plan_id: str | None = None

    reply: str
    actions: list[PlannedActionOut] = Field(default_factory=list)
    rejected: list[RejectedActionOut] = Field(default_factory=list)
    missing: list[MissingItemOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    grounding: Literal["grounded", "not_required", "unverified"] = "not_required"
    step: StepOut | None = None
