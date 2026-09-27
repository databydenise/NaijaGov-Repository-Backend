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
    MAX_HISTORY_TURN_CHARS,
    MAX_HISTORY_TURNS_RENDERED,
    MAX_PAGE_FIELDS,
    MAX_RENDERED_LABEL_CHARS,
    MAX_RENDERED_OPTIONS,
    SNAPSHOT_CLOSE,
    SNAPSHOT_OPEN,
    TOKEN_BUDGET,
    USER_DATA_CLOSE,
    USER_DATA_OPEN,
)
from src.ai.prompt_loader import USER_PLAN
from src.ai.utils import clean, estimate_tokens, mask_value
from src.context.schemas import PageButton, PageField
from src.profiles.constants import PROFILE_FIELDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepView:
    """Where the page sits in its workflow, for the `STEP:` line. Enough to say "1 of 2"."""

    name: str
    index: int
    total: int


@dataclass(frozen=True)
class PlanContext:
    """The assembled context, its estimated size, and what had to be dropped to fit the budget."""

    text: str
    estimated_tokens: int
    truncated: list[str] = field(default_factory=list)


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


def render_page_block(
    step: StepView,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton],
    *,
    required_only: bool = False,
) -> str:
    """
    The `<page_snapshot>` block: the step line, then fields (required first, capped at 80), then
    buttons. `required_only` drops the optional fields — the budget's second overflow stage.
    """
    ordered = _order_fields(fields)
    if required_only:
        ordered = [page_field for page_field in ordered if page_field.required]

    shown = ordered[:MAX_PAGE_FIELDS]
    dropped = len(ordered) - len(shown)

    lines = [SNAPSHOT_OPEN]
    lines.append(f"STEP: {clean(step.name)} ({step.index} of {step.total})")

    lines.append("FIELDS")
    lines.extend(_render_field_line(page_field) for page_field in shown)
    if dropped > 0:
        lines.append(f"  … {dropped} more fields not shown")

    if buttons:
        lines.append("BUTTONS")
        for button in buttons:
            status = "BLOCKED: sensitive" if button.sensitive else "allowed"
            lines.append(f"  {button.field_id} | {clean(button.text, MAX_RENDERED_LABEL_CHARS)} | {status}")

    lines.append(SNAPSHOT_CLOSE)

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


def render_plan_context(
    *,
    step: StepView,
    fields: Sequence[PageField],
    buttons: Sequence[PageButton],
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
    """
    user_data_block = render_user_data_block(profile, chat_values)
    clean_message = clean(message)
    truncated: list[str] = []

    def assemble(page_block: str, history_block: str) -> str:
        return USER_PLAN.format(
            page_block=page_block,
            user_data_block=user_data_block,
            history_block=history_block,
            message=clean_message,
        )

    page_block = render_page_block(step, fields, buttons)
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
            page_block = render_page_block(step, fields, buttons, required_only=True)
            text = assemble(page_block, history_block)

    return PlanContext(text=text, estimated_tokens=estimate_tokens(text), truncated=truncated)
