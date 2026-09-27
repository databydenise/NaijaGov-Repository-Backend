"""
A scripted model, for exercising a turn with no API key and no network.

Every check in `scripts/check_agent.py` runs against this. It is the only way to test the
runner's real subject matter — budgets, retries, tool rounds, the repair pass — because all of
those are *sequences of replies*, and a real model does not produce a chosen sequence on demand.

It lives in `src/ai/` beside the `ModelClient` Protocol it satisfies, not in `scripts/`, for the
reason `tools.py` sits beside `service.py` in `documents/`: a double that drifts from the
interface it doubles is worse than no double, and the two are hard to drift apart from one file
away. Nothing under `src/` imports it at runtime, and it makes no request of any kind.

It records what it was asked, which is half of what the checks assert: that the final call went
out with tools disabled and the schema enforced, that the repair call did not carry the failed
answer back as an assistant turn, and that nothing was called after the budget ran out.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.ai.client import Message, ModelReply, ToolCallRequest
from src.documents.tools import SEARCH_TOOL_NAME


class ScriptExhaustedError(AssertionError):
    """
    The runner made more calls than the script had replies for.

    An `AssertionError` and deliberately *not* a `ModelError`: the runner catches model errors
    and turns them into a failure code, which would quietly convert a wrong script into a
    plausible-looking pass. This has to escape.
    """


@dataclass(frozen=True)
class RecordedCall:
    """One call as the fake received it, reduced to what a check needs to assert."""

    messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    schema_name: str | None
    timeout_seconds: float

    @property
    def tools_offered(self) -> bool:
        return bool(self.tool_names)

    @property
    def schema_enforced(self) -> bool:
        return self.schema_name is not None

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(str(message.get("role", "")) for message in self.messages)

    def text_at(self, index: int) -> str:
        """The content of one message, for asserting what a repair instruction said."""
        content = self.messages[index].get("content")

        return content if isinstance(content, str) else ""


@dataclass
class FakeModelClient:
    """
    A `ModelClient` that replays a fixed script.

    Each entry is either a `ModelReply` to return or an exception to raise, in order. A script
    of one reply and a client that is called twice raises `ScriptExhaustedError` rather than
    repeating the last answer, so a check cannot pass by accident.
    """

    script: Sequence[ModelReply | Exception]
    model: str = "fake-model"
    calls: list[RecordedCall] = field(default_factory=list)
    _index: int = 0

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        timeout_seconds: float,
    ) -> ModelReply:
        self.calls.append(
            RecordedCall(
                messages=tuple(messages),
                tool_names=tuple(
                    str(tool.get("function", {}).get("name", "")) for tool in tools
                ),
                schema_name=schema_name if response_schema is not None else None,
                timeout_seconds=timeout_seconds,
            ),
        )

        if self._index >= len(self.script):
            message = (
                f"the runner made {len(self.calls)} model calls; "
                f"the script has {len(self.script)}"
            )
            raise ScriptExhaustedError(message)

        step = self.script[self._index]
        self._index += 1

        if isinstance(step, Exception):
            raise step

        return step

    @property
    def call_count(self) -> int:
        return len(self.calls)


def answer(
    payload: dict[str, Any] | str,
    *,
    prompt_tokens: int = 1200,
    completion_tokens: int = 200,
) -> ModelReply:
    """
    A reply carrying a structured answer. A dict is serialised; a string is sent as it is, which
    is how a malformed answer is scripted.
    """
    text = payload if isinstance(payload, str) else json.dumps(payload)

    return ModelReply(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def searches(
    *queries: str | tuple[str, dict[str, Any] | str],
    prompt_tokens: int = 1200,
    completion_tokens: int = 60,
) -> ModelReply:
    """
    A reply asking for one or more retrieval calls.

    A bare string is that query with default arguments; a `(name, arguments)` pair scripts
    anything else, including a tool the runner does not offer and arguments it must refuse.
    """
    calls: list[ToolCallRequest] = []

    for index, item in enumerate(queries):
        if isinstance(item, str):
            name, arguments = SEARCH_TOOL_NAME, {"query": item}
        else:
            name, arguments = item

        calls.append(
            ToolCallRequest(
                id=f"call_{index + 1}",
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            ),
        )

    return ModelReply(
        tool_calls=tuple(calls),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="tool_calls",
    )
