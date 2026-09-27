"""
`/results` request and response models.

The request is the narrowest one this service has, and it is narrow on purpose. It carries an
action id, a field id, a status and a reason code — four things, none of which can hold what a
citizen typed. `extra="forbid"` would already refuse a fifth, but `_no_content_keys` names the
ones that matter in its own words, because the mistake this guards against is not a typo. It is
someone reasonably thinking it would be useful to know what went into the form.

The response is a fixed shape whether or not anything was recorded. An unacknowledged report
comes back with the same keys, zero counts and a default hint, so the panel renders one summary
card and never branches on "did the backend know about this run".
"""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.constants import SAFE_MESSAGE_MARKER
from src.context.constants import MAX_FIELD_ID_LENGTH
from src.results.constants import (
    ABORT_REASONS,
    FORBIDDEN_RESULT_KEYS,
    MAX_ACTION_ID_CHARS,
    MAX_ELAPSED_MS,
    MAX_PLAN_ID_CHARS,
    MAX_REASON_CHARS,
    MAX_RESULTS,
    RESULT_REASONS,
    RESULT_STATUSES,
)

ResultStatus = Literal["ok", "changed", "failed", "rejected", "cancelled"]

ActionId = Annotated[str, Field(min_length=1, max_length=MAX_ACTION_ID_CHARS)]
FieldId = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_ID_LENGTH)]
ReasonCode = Annotated[str, Field(max_length=MAX_REASON_CHARS)]


class ReportModel(BaseModel):
    """Base for everything in a report: unknown keys are an error, not noise."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _refuse_content_keys(data: object) -> object:
    """
    Refuse an object carrying what was written or what the page called it.

    `extra="forbid"` rejects these anyway, with Pydantic's generic wording and without saying why
    it matters. This says so in its own words, and names only the key — never what was in it.
    """
    if not isinstance(data, dict):
        return data

    offending = sorted(FORBIDDEN_RESULT_KEYS.intersection(map(str, data)))

    if offending:
        # Marked safe so the handler shows it verbatim. Everything interpolated here is a key
        # name from our own constant, never a submitted value.
        message = (
            f"{SAFE_MESSAGE_MARKER} A result may not carry {', '.join(offending)}. "
            f"This service records which field was touched and how it went, never what was "
            f"written into it."
        )
        raise ValueError(message)

    return data


class ResultEntry(ReportModel):
    """
    What happened to one action.

    `action_id` is the positional id this service minted in the plan response (`a1`, `a2`), and it
    is the only part of this the service trusts: the action's type and the field it targets are
    read back from the plan rather than from here, so a mismatched pair cannot write a row that
    disagrees with what was actually approved.

    `reason` is required for every status but `ok` — a `failed` with no reason is a row that
    records that something went wrong and loses the only part worth knowing.
    """

    action_id: ActionId
    field_id: FieldId
    status: ResultStatus
    reason: ReasonCode | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_content_keys(cls, data: object) -> object:
        return _refuse_content_keys(data)

    @model_validator(mode="after")
    def _check_reason(self) -> "ResultEntry":
        if self.reason and self.reason not in RESULT_REASONS:
            # The code, not the value: this is a fixed vocabulary both repos share, so naming the
            # rejected code is naming one of our own constants back.
            message = f"{SAFE_MESSAGE_MARKER} That is not a reason code this service knows."
            raise ValueError(message)

        if self.status != "ok" and not self.reason:
            message = f"{SAFE_MESSAGE_MARKER} A result that is not 'ok' must carry a reason code."
            raise ValueError(message)

        return self


class CheckpointReport(ReportModel):
    """
    Where a run stopped because the page asked for something only the user can give.

    `reason` is a short kind — "password", "otp" — and `after_index` is how many actions had
    already run. Both are positions and categories; neither is content.
    """

    reason: Annotated[str, Field(min_length=1, max_length=MAX_REASON_CHARS)]
    after_index: Annotated[int, Field(ge=0, le=MAX_RESULTS)]

    @model_validator(mode="before")
    @classmethod
    def _no_content_keys(cls, data: object) -> object:
        return _refuse_content_keys(data)


class ResultsRequest(ReportModel):
    """
    What the extension sends once a run has finished, or stopped.

    Every field is history. Nothing here asks permission for anything, which is why an unknown
    plan is answered rather than refused — see `service.py`.
    """

    session_id: uuid.UUID
    plan_id: Annotated[str, Field(min_length=1, max_length=MAX_PLAN_ID_CHARS)]

    results: list[ResultEntry] = Field(default_factory=list, max_length=MAX_RESULTS)

    checkpoint: CheckpointReport | None = None

    # Why the whole batch stopped, when it did. A checkpoint is reported above instead: it is the
    # one stop that means the safety layer worked rather than that something broke.
    aborted: Literal["stale_page", "navigated", "timeout"] | None = None

    elapsed_ms: Annotated[int, Field(ge=0, le=MAX_ELAPSED_MS)] = 0

    @model_validator(mode="after")
    def _check_abort(self) -> "ResultsRequest":
        if self.aborted is not None and self.aborted not in ABORT_REASONS:  # pragma: no cover
            # Unreachable through the Literal above, and kept so the enum and the constant cannot
            # drift apart silently if one of them is ever widened alone.
            message = f"{SAFE_MESSAGE_MARKER} That is not an abort reason this service knows."
            raise ValueError(message)

        return self


class RunTotals(BaseModel):
    """This run's outcome, counted by status. Every key present, so the panel can render a row."""

    ok: int = 0
    changed: int = 0
    failed: int = 0
    rejected: int = 0
    cancelled: int = 0


class NextHintOut(BaseModel):
    """The code the panel may branch on, and the sentence it prints verbatim."""

    code: str
    message: str


class StepOut(BaseModel):
    """Where the run happened. Absent for a page the registry does not know."""

    id: str
    index: int
    total: int
    is_final: bool


class ResultsResponse(BaseModel):
    """
    What the panel needs to close out a run.

    `acknowledged: false` means this service has no record of the plan — expired, unknown, or
    already reported by an earlier call whose answer was lost. It is not an error and the shape
    does not change: zero counts and a hint that claims nothing.
    """

    acknowledged: bool
    recorded: int = 0
    next_hint: NextHintOut
    step: StepOut | None = None
    run: RunTotals = Field(default_factory=RunTotals)


def statuses_are_in_sync() -> bool:
    """
    Whether `ResultStatus` and `RESULT_STATUSES` still agree.

    A belt-and-braces check for a list that exists twice — once as a typing `Literal` the request
    is parsed against, once as a tuple the database CHECK mirrors. Called by the check script
    rather than at import: a mismatch is a development-time mistake, and failing a boot over it
    would take an instance down for something no request can trigger.
    """
    return set(ResultStatus.__args__) == set(RESULT_STATUSES)
