"""
What the guard returns.

Frozen dataclasses, not pydantic models, for the same reason `documents/schemas.py` uses them:
nothing here parses untrusted input — everything has already been through `PlanResponse` — and
these are built on the `/plan` path. P4 maps them onto its response model.

The shape is deliberately not "a list of actions". An approved action carries the literal value
*and* where that value came from, so the fill preview can show the user "Email ← profile.email"
before anything is written; a rejected action carries a code for us and a sentence for them. A
plan the panel cannot explain is a plan the user cannot check, and an unexplained fill on a
government form is the failure this whole fragment exists to prevent.
"""

from dataclasses import dataclass, field

from src.ai.schemas import Citation, MissingItem
from src.guard.constants import GroundingVerdict, RejectionCode


@dataclass(frozen=True)
class ApprovedAction:
    """
    One action the extension may apply, with its value already substituted.

    `source` is the reference the value came from, as `profile.email` or `chat.lga`, and is
    None for an action that writes nothing. It is the provenance the preview shows: every
    written value traces to the profile or to something the user typed, and an action whose
    source could not be named never reaches this list.

    `suspicious` is the one soft outcome in the guard — the value resolved, the field accepted
    it, but the key and the label disagree (Section 3). Approved, and marked, so the preview
    can put a warning beside that row rather than dropping a legitimate fill.
    """

    type: str
    field_id: str | None = None
    value: str | None = None
    source: str | None = None
    checked: bool | None = None
    reason: str | None = None
    note: str | None = None
    suspicious: bool = False


@dataclass(frozen=True)
class RejectedAction:
    """
    One action that did not survive, with a code for the log and a sentence for the user.

    Reported rather than silently removed: "1 suggestion was blocked (password field)" reads as
    care, while silence reads as an unreliable tool and hides our own bugs.
    """

    type: str
    code: RejectionCode
    message: str
    field_id: str | None = None


@dataclass(frozen=True)
class GuardReport:
    """
    The summary for the log line and the response.

    Counts, codes and field ids only. No value, no label, no reply text — a log line that
    carries a profile value is a privacy incident whatever the log level says.
    """

    approved: int = 0
    rejected: int = 0
    suspicious: int = 0
    rejection_codes: dict[str, int] = field(default_factory=dict)
    rejected_field_ids: tuple[str, ...] = ()
    grounding: GroundingVerdict = "not_required"
    citations_verified: int = 0
    citations_dropped: int = 0
    missing: int = 0
    missing_added: int = 0
    repair_requested: bool = False
    reply_replaced: bool = False


@dataclass(frozen=True)
class GuardedPlan:
    """
    The whole of what the guard hands P4.

    `repair_requested` is the guard's only instruction to its caller: the reply made a factual
    claim nothing retrieved supports, so ask the model once more with the failure fed back.
    Actions are already final either way — a wrong sentence does not invalidate a correct fill.
    """

    reply: str
    approved: tuple[ApprovedAction, ...] = ()
    rejected: tuple[RejectedAction, ...] = ()
    citations: tuple[Citation, ...] = ()
    missing: tuple[MissingItem, ...] = ()
    grounding: GroundingVerdict = "not_required"
    repair_requested: bool = False
    reply_replaced: bool = False

    # Every `chat.*` value this plan could resolve against: what the session already held, plus
    # what the model extracted from this turn's message, `null`s skipped. Carried out of the
    # guard because the caller has to store it and must store *exactly* this — a fill approved
    # because `chat.lga` resolved, against a session that then saved a different `lga`, is a
    # preview the user accepted and a context that disagrees with it on the next turn.
    chat_values: dict[str, str] = field(default_factory=dict)

    report: GuardReport = field(default_factory=GuardReport)
