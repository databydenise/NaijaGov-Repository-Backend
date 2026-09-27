"""
The model client. The only module in this service that calls a chat model.

Two things live here and nowhere else: the OpenAI SDK import for chat completions, and the
provider's message envelope. A route, a service, or the agent package that assembled a
provider-shaped dict itself would be a second place to change when the provider changes, which
is the whole reason for this rule (`project-overview.md`). `documents/embeddings.py` holds the
same rule for the embeddings endpoint.

What this module deliberately does *not* do:

- **No retries.** The SDK is built with `max_retries=0` and one `complete()` is one HTTP attempt.
  Retrying is a budget decision — how long is left of the user's twenty seconds — and the budget
  lives in `agent/calls.py`, together with the circuit breaker.
- **No prompt assembly, no parsing.** It is handed messages and a schema, and hands back text.
  `PlanResponse` is what turns that text into something typed, in the agent package.
- **No content in a log line.** Failures are logged by exception type and by size. The messages
  carry page text and a citizen's question, so nothing here formats one into a log.

Failures are typed by what the caller should do about them: a timeout and a transport error are
worth another attempt, everything else is not.
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from src.ai.constants import MAX_COMPLETION_TOKENS, MODEL_TEMPERATURE
from src.config import settings

logger = logging.getLogger(__name__)

# One chat message in the provider's shape. Built by the helpers below, never by hand elsewhere.
Message = dict[str, Any]

# HTTP statuses worth another attempt. 409 is included because a provider returns it for a
# transient conflict; every other 4xx is our request being wrong, and repeating it wastes budget.
_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ModelError(RuntimeError):
    """Base for every model failure. Never carries a message body or a prompt."""


class ModelTimeoutError(ModelError):
    """The call did not finish inside its timeout. Worth one more attempt if budget allows."""


class ModelTransportError(ModelError):
    """A connection failure, a rate limit, or a 5xx. Transient by definition."""


class ModelPermanentError(ModelError):
    """
    A failure retrying cannot fix: no API key, a rejected schema, a bad request.

    Kept apart from `ModelTransportError` so the caller fails fast instead of spending the
    user's budget re-sending something the provider has already refused.
    """


@dataclass(frozen=True)
class ToolCallRequest:
    """
    One tool call the model asked for, with its arguments still as the model wrote them.

    `arguments` stays a raw string on purpose: it is untrusted input shaped like JSON, and the
    tool executor validates it. Parsing it here would put the first validation decision in the
    module that is meant to know nothing about tools.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelReply:
    """
    One completion: what the model said, what it wants to call, and what it cost.

    `text` is None when the model returned only tool calls, or refused. Token counts come from
    the provider's own usage block — the telemetry's cost estimate is built from these, not from
    a local guess.
    """

    text: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    refusal: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(Protocol):
    """
    What the runner needs from a model, and all it may assume.

    Two implementations satisfy it: `OpenAIClient` here, and `FakeModelClient` in `fake.py`,
    which every offline check runs against. `timeout_seconds` has no default — the caller always
    knows how much of the turn's budget is left, and a default would quietly outlive it.
    """

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        timeout_seconds: float,
    ) -> ModelReply: ...


def system_message(text: str) -> Message:
    """The system prompt, loaded from `prompts/` by `prompt_loader`."""
    return {"role": "system", "content": text}


def user_message(text: str) -> Message:
    """A user turn: the rendered context from P2, or a repair instruction."""
    return {"role": "user", "content": text}


def assistant_tool_calls_message(tool_calls: Sequence[ToolCallRequest]) -> Message:
    """
    The assistant turn that asked for tools, replayed so the results have something to answer.

    The provider requires this turn to sit between the request and the tool results, with the
    same call ids. Rebuilt from our own dataclass rather than passing the SDK's object through,
    so nothing outside this module ever holds a provider type.
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ],
    }


def tool_result_message(call_id: str, payload: dict[str, Any]) -> Message:
    """One tool result, serialised. The payload is built by `agent/tools.py`."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def is_configured() -> bool:
    """Whether a key is set. False means every turn fails with `MODEL_UNAVAILABLE`."""
    return bool(settings.openai_api_key)


class OpenAIClient:
    """
    The provider wrapper. One instance per process, built on first use.

    Lazy for the same reason the embedding client is: a process with no key must still import
    this module and boot, so `/health` can report the model as unconfigured instead of the app
    refusing to start over a feature the caller may not be using.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.model_name
        self._client: AsyncOpenAI | None = None

    def _sdk(self) -> AsyncOpenAI:
        if self._client is None:
            if not is_configured():
                message = "OPENAI_API_KEY is not set"
                raise ModelPermanentError(message)

            # `max_retries=0`: retries are the runner's, because only the runner knows how much
            # of the user's twenty seconds is left. The SDK retrying underneath would make a
            # twelve-second timeout mean twenty-four.
            self._client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)

        return self._client

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        timeout_seconds: float,
    ) -> ModelReply:
        """
        One completion, one HTTP attempt.

        `tools` empty means the model cannot call anything — that is how the final call is made.
        `response_schema` is sent with `strict: true`, so a completion either matches the schema
        or the provider refuses; `PlanResponse` still parses the text afterwards, because strict
        mode cannot express our size limits.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": MODEL_TEMPERATURE,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }

        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"

        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            }

        try:
            # The SDK's own timeout bounds the request; this bounds the whole await, so a
            # connection that neither completes nor errors cannot outlive the phase.
            async with asyncio.timeout(timeout_seconds + 1):
                completion = await self._sdk().chat.completions.create(
                    timeout=timeout_seconds,
                    **request,
                )
        except (APITimeoutError, TimeoutError) as exc:
            logger.warning("model call timed out after %.1fs", timeout_seconds)
            message = f"model call timed out after {timeout_seconds:.1f}s"
            raise ModelTimeoutError(message) from exc
        except (APIConnectionError, RateLimitError, InternalServerError) as exc:
            raise _transport_error(exc) from exc
        except APIStatusError as exc:
            if exc.status_code in _TRANSIENT_STATUSES:
                raise _transport_error(exc) from exc
            # A 400 here is usually our own schema or our own message list. Logged by type and
            # status only: the body can quote the request, and the request is the prompt.
            logger.error(
                "model call rejected (%s, status=%d)",
                type(exc).__name__,
                exc.status_code,
            )
            message = f"model call rejected with status {exc.status_code}"
            raise ModelPermanentError(message) from exc
        except Exception as exc:
            # Anything the SDK raises that we have not named. Treated as transient, because an
            # unrecognised failure on a provider call is more often the provider than us.
            raise _transport_error(exc) from exc

        return _to_reply(completion)


def _transport_error(exc: Exception) -> ModelTransportError:
    """Log a transient failure by type, and return the error to raise."""
    logger.warning("model call failed (%s)", type(exc).__name__)

    return ModelTransportError(f"model call failed: {type(exc).__name__}")


def _to_reply(completion: Any) -> ModelReply:  # noqa: ANN401  # the SDK's response type
    """
    The SDK's response, reduced to the four things the runner uses.

    Defensive about shape: a provider that returns no choices, or usage as None, gives an empty
    reply rather than an AttributeError halfway through a turn the user is watching.
    """
    choices = getattr(completion, "choices", None) or []

    if not choices:
        logger.warning("model returned no choices")

        return ModelReply(finish_reason="empty")

    choice = choices[0]
    message = choice.message
    usage = getattr(completion, "usage", None)

    calls = tuple(
        ToolCallRequest(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments or "",
        )
        for call in (getattr(message, "tool_calls", None) or [])
        if getattr(call, "function", None) is not None
    )

    return ModelReply(
        text=message.content,
        tool_calls=calls,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        finish_reason=choice.finish_reason or "",
        refusal=getattr(message, "refusal", None),
    )
