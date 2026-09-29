"""
The context renderer: a page snapshot in, a prompt string out.

Pure functions only — no database, no HTTP, no clock, no randomness — so a rendering is fully
determined by its inputs and can be pinned by a golden file. Two jobs run through here:

- Turn the structure of a page into text the model can reason about, with every field and button
  shown (blocked ones included, so the model can explain a refusal rather than guess).
- Keep the user's real values out of that text. The profile block shows keys with a *masked*
  preview; the real value never appears, which is the property `scripts/check_ai_contract.py`
  proves. Values the user typed in this chat are the one exception — the model already saw them —
  and are shown in full so the fill it was just asked for still works.

This module owns two of the three injection defences: escaping the fence delimiters and stripping
control characters (here), and the system prompt's "page content is data" instruction (in
`prompts/`). The third — validating what the model says against the real snapshot — is P3's guard.
None is sufficient alone.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.ai.constants import (
    DEFAULT_EXPLAIN_QUESTION,
    MAX_HISTORY_TURN_CHARS,
    MAX_HISTORY_TURNS_RENDERED,
    MAX_PAGE_FIELDS,
    MAX_RENDERED_CHUNK_CHARS,
    MAX_RENDERED_CHUNKS,
    MAX_RENDERED_HREF_CHARS,
    MAX_RENDERED_LABEL_CHARS,
    MAX_RENDERED_NEARBY_CHARS,
    MAX_RENDERED_OPTIONS,
    MAX_RENDERED_QUESTION_CHARS,
    MAX_RENDERED_STEP_LABELS,
    MAX_RENDERED_STEPS,
    OFFICIAL_SOURCES_CLOSE,
    OFFICIAL_SOURCES_OPEN,
    SNAPSHOT_CLOSE,
    SNAPSHOT_OPEN,
    TOKEN_BUDGET,
    USER_DATA_CLOSE,
    USER_DATA_OPEN,
    WORKFLOW_CLOSE,
    WORKFLOW_OPEN,
)
from src.ai.prompt_loader import USER_EXPLAIN, USER_PLAN
from src.ai.utils import clean, estimate_tokens, mask_value
from src.context.schemas import PageButton, PageField, PageLink
from src.documents.schemas import RetrievedChunk
from src.profiles.constants import PROFILE_FIELDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepView:
    """Where the page sits in its workflow, for the `STEP:` line. Enough to say "1 of 2"."""

    name: str
    index: int
    total: int


@dataclass(frozen=True)
class WorkflowStepLine:
    """One step of the workflow as the registry holds it, for the `<workflow>` block."""

    name: str
    index: int
    is_final: bool
    field_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowView:
    """
    The whole workflow, from our own registry rather than from the page.

    This is what makes "where do I start?", "what comes next?" and "what will I need?" answerable
    without reading the DOM — which matters most on the pages where the DOM says least. A portal
    landing page has no form on it, and before this the only honest answer available to the model
    was to describe its own empty input back to the user.

    `current_index` is the step the user is on, matched by `index`. None for a page we placed in a
    workflow but not on a step, which renders the list with nothing marked rather than guessing.
    """

    name: str
    agency: str
    steps: tuple[WorkflowStepLine, ...] = ()
    current_index: int | None = None


@dataclass(frozen=True)
class PlanContext:
    """The assembled context, its estimated size, and what had to be dropped to fit the budget."""

    text: str
    estimated_tokens: int
    truncated: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExplainContext:
    """
    The assembled `/explain` context and its estimated size.

    No `truncated` list, because there is nothing to drop: one field, at most three capped chunks,
    a capped question and no history is a small prompt by construction. If that ever stops being
    true the answer is a smaller chunk cap, not an overflow stage that silently removes the
    evidence the answer has to be grounded in.
    """

    text: str
    estimated_tokens: int


def _order_fields(fields: Sequence[PageField]) -> list[PageField]:
    """Required fields first, document order within each group (a stable sort keeps it)."""
    return sorted(fields, key=lambda page_field: not page_field.required)


def _render_field_line(page_field: PageField) -> str:
    """One `  id | label | type | status [| options: …]` line."""
    label = clean(page_field.label, MAX_RENDERED_LABEL_CHARS)
    field_type = clean(page_field.type, MAX_RENDERED_LABEL_CHARS) or "text"

    if page_field.sensitive:
        status = "BLOCKED: sensitive"
    elif page_field.required:
        status = "required"
    else:
        status = "optional"

    parts = [page_field.field_id, label, field_type, status]

    if page_field.options:
        shown = [clean(option, MAX_RENDERED_LABEL_CHARS) for option in page_field.options[:MAX_RENDERED_OPTIONS]]
        rendered = ", ".join(shown)
        if len(page_field.options) > MAX_RENDERED_OPTIONS:
            rendered += ", …"
        parts.append(f"options: {rendered}")

    return "  " + " | ".join(parts)


def _render_link_line(link: PageLink) -> str:
    """One `  id | text | href | status` line. `status` is never "allowed": a link is not ours."""
    parts = [
        link.field_id,
        clean(link.text, MAX_RENDERED_LABEL_CHARS),
        clean(link.href, MAX_RENDERED_HREF_CHARS) or "(no address)",
    ]

    if link.sensitive:
        parts.append("BLOCKED: sensitive")
    elif link.external:
        parts.append("leaves this site")
    else:
        parts.append("same site")

    return "  " + " | ".join(parts)


def render_page_block(  # noqa: PLR0913  # the snapshot's own parts, plus one rendering switch
    step: StepView,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton],
    links: Sequence[PageLink] = (),
    *,
    required_only: bool = False,
) -> str:
    """
    The `<page_snapshot>` block: the step line, then fields (required first, capped at 80), then
    buttons, then navigation links. `required_only` drops the optional fields — the budget's
    second overflow stage.

    Links are rendered under their own heading, and the heading says they cannot be clicked. That
    is the whole reason they are not folded in with the buttons: a landing page has no form and no
    buttons, so without this section the model is shown an empty page and can only answer by
    asking the user to go and look — which is the product failing at its job. With it, the model
    can name the route the user needs. It still may not take that route for them.
    """
    ordered = _order_fields(fields)
    if required_only:
        ordered = [page_field for page_field in ordered if page_field.required]

    shown = ordered[:MAX_PAGE_FIELDS]
    dropped = len(ordered) - len(shown)

    lines = [SNAPSHOT_OPEN]
    lines.append(f"STEP: {clean(step.name)} ({step.index} of {step.total})")

    lines.append("FIELDS")
    if shown:
        lines.extend(_render_field_line(page_field) for page_field in shown)
    else:
        # Said in words rather than left as a bare heading. A landing page has no form on it, and
        # an empty list under a heading is the shape that produced "the current page snapshot
        # shows no fields or buttons" as an answer to a citizen — the model reporting its input
        # because nothing told it that an empty form is an ordinary page rather than a fault.
        lines.append("  (none — this page has no form on it)")
    if dropped > 0:
        lines.append(f"  … {dropped} more fields not shown")

    if buttons:
        lines.append("BUTTONS")
        for button in buttons:
            status = "BLOCKED: sensitive" if button.sensitive else "allowed"
            lines.append(f"  {button.field_id} | {clean(button.text, MAX_RENDERED_LABEL_CHARS)} | {status}")

    if links:
        lines.append("LINKS (navigation — name one in your reply, never click it yourself)")
        lines.extend(_render_link_line(link) for link in links)

    lines.append(SNAPSHOT_CLOSE)

    return "\n".join(lines)


def _render_step_line(step: WorkflowStepLine, *, is_current: bool) -> list[str]:
    """One step as one or two lines: its position and name, then what it asks for."""
    marker = "  ← the page you are on" if is_current else ""
    if step.is_final and not is_current:
        marker = "  (the last step)"

    lines = [f"  {step.index}. {clean(step.name, MAX_RENDERED_LABEL_CHARS)}{marker}"]

    if step.field_labels:
        shown = [
            clean(label, MAX_RENDERED_LABEL_CHARS)
            for label in step.field_labels[:MAX_RENDERED_STEP_LABELS]
        ]
        rendered = ", ".join(shown)
        if len(step.field_labels) > MAX_RENDERED_STEP_LABELS:
            rendered += ", …"
        lines.append(f"     asks for: {rendered}")

    return lines


def render_workflow_block(workflow: WorkflowView | None) -> str:
    """
    The `<workflow>` block: which process this is, and every step of it in order.

    Deliberately **outside** `<page_snapshot>`. Everything in that block is text copied from a web
    page and is declared untrusted; this is the registry we seeded and checked in ourselves, and
    the model is told it may rely on it. Two kinds of data with opposite trust levels must not
    share a fence — and `ai/utils.clean` escapes this block's delimiters in page text so a label
    cannot forge one.

    What it is *not* is a source. A step name is a fact about the portal's own form; a fee, a
    processing time or a document requirement is a claim about government, and those still come
    only from retrieval. The system prompt says so, and the guard's grounding check is unchanged.
    """
    if workflow is None or not workflow.steps:
        return ""

    lines = [
        WORKFLOW_OPEN,
        f"{clean(workflow.name)} — {clean(workflow.agency)}",
        f"STEPS ({len(workflow.steps)} in all, from our own records, not read from this page):",
    ]

    for step in workflow.steps[:MAX_RENDERED_STEPS]:
        lines.extend(
            _render_step_line(step, is_current=step.index == workflow.current_index),
        )

    if len(workflow.steps) > MAX_RENDERED_STEPS:
        lines.append(f"  … {len(workflow.steps) - MAX_RENDERED_STEPS} more steps not shown")

    lines.append(WORKFLOW_CLOSE)

    return "\n".join(lines)


def render_user_data_block(
    profile: Mapping[str, str | None],
    chat_values: Mapping[str, str],
) -> str:
    """
    The `<user_data>` block: available profile keys with masked previews, a MISSING list, and any
    values from this chat shown in full. Profile keys are listed in their canonical order, so a
    rendering is stable regardless of how the mapping was built.
    """
    available: list[str] = []
    missing: list[str] = []

    for key in PROFILE_FIELDS:
        value = profile.get(key)
        if value:
            available.append(f"  profile.{key} = \"{mask_value(value)}\"")
        else:
            missing.append(f"profile.{key}")

    lines = [USER_DATA_OPEN, "AVAILABLE (reference these, never write the value yourself):"]
    lines.extend(available if available else ["  (nothing on file yet)"])

    if missing:
        lines.append(f"MISSING: {', '.join(missing)}")

    if chat_values:
        lines.append("FROM THIS CHAT:")
        for key in chat_values:
            lines.append(f"  chat.{key} = \"{clean(chat_values[key], MAX_HISTORY_TURN_CHARS)}\"")

    lines.append(USER_DATA_CLOSE)

    return "\n".join(lines)


def render_history_block(history: Sequence[Mapping[str, Any]]) -> str:
    """
    The last few turns as `USER:` / `ASSISTANT:` lines, each capped at 500 characters. Older turns
    are dropped, not summarised. A turn shape is `{"role": "user"|"assistant", "content": str}`;
    anything else in the mapping is ignored, and a turn with an unknown role is skipped.
    """
    rendered: list[str] = []

    for turn in history:
        role = str(turn.get("role", "")).lower()
        if role not in ("user", "assistant"):
            continue

        content = clean(str(turn.get("content", "")), MAX_HISTORY_TURN_CHARS)
        if not content:
            continue

        rendered.append(f"{role.upper()}: {content}")

    recent = rendered[-MAX_HISTORY_TURNS_RENDERED:]
    if not recent:
        return ""

    return "\n".join(["CONVERSATION SO FAR:", *recent])


def render_plan_context(  # noqa: PLR0913  # one keyword per part of the rendered context
    *,
    step: StepView,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton],
    links: Sequence[PageLink] = (),
    workflow: WorkflowView | None = None,
    profile: Mapping[str, str | None],
    chat_values: Mapping[str, str],
    history: Sequence[Mapping[str, Any]],
    message: str,
) -> PlanContext:
    """
    Assemble the full `/plan` user message from the `user_plan` template and keep it under the
    token budget. On overflow, drop in the order the spec sets — history first, then non-required
    fields — logging each drop, because truncation correlates with worse answers. Retrieved chunks
    reach the model through the search tool result, not this string, so their budgeting is P4's.

    The `<workflow>` block is never dropped. It is a few hundred characters whatever the page is,
    and it is the only part of this context that survives a page with nothing on it — dropping it
    to make room for form fields would take the answer away from precisely the pages that have no
    form to describe.
    """
    user_data_block = render_user_data_block(profile, chat_values)
    workflow_block = render_workflow_block(workflow)
    clean_message = clean(message)
    truncated: list[str] = []

    def assemble(page_block: str, history_block: str) -> str:
        return USER_PLAN.format(
            workflow_block=workflow_block,
            page_block=page_block,
            user_data_block=user_data_block,
            history_block=history_block,
            message=clean_message,
        )

    page_block = render_page_block(step, fields, buttons, links)
    history_block = render_history_block(history)
    text = assemble(page_block, history_block)

    if estimate_tokens(text) > TOKEN_BUDGET and history_block:
        truncated.append("history")
        logger.warning(
            "plan context over budget; dropping history (fields=%d buttons=%d)",
            len(fields),
            len(buttons),
        )
        history_block = ""
        text = assemble(page_block, history_block)

    if estimate_tokens(text) > TOKEN_BUDGET:
        required = sum(1 for page_field in fields if page_field.required)
        if required < len(fields):
            truncated.append("non_required_fields")
            logger.warning(
                "plan context over budget; dropping %d non-required fields",
                len(fields) - required,
            )
            page_block = render_page_block(step, fields, buttons, links, required_only=True)
            text = assemble(page_block, history_block)

    return PlanContext(text=text, estimated_tokens=estimate_tokens(text), truncated=truncated)


def render_explain_field_block(
    step: StepView | None,
    page_field: PageField,
    nearby_text: str = "",
) -> str:
    """
    The `<page_snapshot>` block for one field: the step line, that field, and the text beside it.

    One field rather than the page, which is the whole difference between this and
    `render_page_block`: `/explain` answers about a single control, and showing the model forty
    others invites it to explain the form instead. `nearby_text` is the portal's own instruction
    next to the field — the most useful sentence on the page and the least trustworthy, so it is
    cleaned and capped like every other piece of page text.
    """
    lines = [SNAPSHOT_OPEN]

    if step is not None:
        lines.append(f"STEP: {clean(step.name)} ({step.index} of {step.total})")

    lines.append("FIELD")
    lines.append(_render_field_line(page_field))

    if nearby_text:
        lines.append(f"NEARBY TEXT: {clean(nearby_text, MAX_RENDERED_NEARBY_CHARS)}")

    lines.append(SNAPSHOT_CLOSE)

    return "\n".join(lines)


def render_sources_block(chunks: Sequence[RetrievedChunk]) -> str:
    """
    The `<official_sources>` block: the chunks retrieved for this field, with their provenance.

    `chunk_id` is on every entry because it is what a citation names, and the guard drops a
    citation whose id was not retrieved — so a chunk rendered without its id is a chunk the model
    cannot legitimately cite. `checked` is the corpus's own ingest date, not today's.

    An empty list renders a block that says so. `/explain` short-circuits before the model call
    when nothing was retrieved, so this is a fallback rather than the path — and a block reading
    "no official sources" is a better fallback than no block at all, which reads as a prompt bug.
    """
    if not chunks:
        return "\n".join(
            [
                OFFICIAL_SOURCES_OPEN,
                "(no official source covers this field)",
                OFFICIAL_SOURCES_CLOSE,
            ],
        )

    lines = [OFFICIAL_SOURCES_OPEN]

    for chunk in chunks[:MAX_RENDERED_CHUNKS]:
        checked = chunk.ingested_at.date().isoformat() if chunk.ingested_at else "unknown"
        lines.append(
            f"[chunk_id {chunk.chunk_id}] {clean(chunk.title, MAX_RENDERED_LABEL_CHARS)} "
            f"| {clean(chunk.source_url, MAX_RENDERED_LABEL_CHARS)} | checked {checked}",
        )
        lines.append(f"  {clean(chunk.content, MAX_RENDERED_CHUNK_CHARS)}")

    lines.append(OFFICIAL_SOURCES_CLOSE)

    return "\n".join(lines)


def render_explain_context(
    *,
    step: StepView | None,
    page_field: PageField,
    chunks: Sequence[RetrievedChunk],
    nearby_text: str = "",
    question: str = "",
) -> ExplainContext:
    """
    Assemble the `/explain` user message: one field, the sources retrieved for it, and the question.

    What is deliberately absent is the `<user_data>` block. `/plan` renders masked profile keys
    because it has to write values into the page; `/explain` writes nothing, so the model is given
    none of the user's data at all — not a key, not a mask. That is a stronger guarantee than the
    prompt's "do not read back stored values", because there is nothing there to read back.
    """
    return _sized(
        USER_EXPLAIN.format(
            page_block=render_explain_field_block(step, page_field, nearby_text),
            sources_block=render_sources_block(chunks),
            question=clean(question, MAX_RENDERED_QUESTION_CHARS) or DEFAULT_EXPLAIN_QUESTION,
        ),
    )


def _sized(text: str) -> ExplainContext:
    """The rendered text with its token estimate, so callers cannot forget to measure it."""
    return ExplainContext(text=text, estimated_tokens=estimate_tokens(text))

