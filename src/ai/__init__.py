"""
The AI contract: what the model is told, and what it is allowed to say back.

Everything here is deterministic. `context.py` turns a page snapshot into text, `schemas.py`
defines the one shape a response may take, and `prompts/` holds the instructions as files so
a prompt change reads as a prompt change in a diff. No module here calls a model — that is P4,
and only the client module added there may import the provider SDK.

The idea that shapes the rest: **the model never receives a real profile value and never emits
one.** It works with keys (`profile.email`), and the guard substitutes the real value after
validation (P3). That single constraint kills the invented-value failure at its root rather
than catching it downstream.

Note: `project-overview.md` still names Gemini and `ai/client.py`. The provider is OpenAI
(strict structured outputs); that line is stale and should be corrected when P4 adds the
client. The one-module rule stands regardless: only the client will import the SDK.
"""
