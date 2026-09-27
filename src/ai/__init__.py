"""
The AI contract: what the model is told, and what it is allowed to say back.

`context.py` turns a page snapshot into text, `schemas.py` defines the one shape a response
may take, and `prompts/` holds the instructions as files so a prompt change reads as a prompt
change in a diff. All of that is deterministic and testable without a model.

`client.py` is the exception and the only one: it holds the OpenAI SDK import for chat
completions and the provider's message envelope, and no other module in this service may import
either. `fake.py` is the scripted double that satisfies the same interface, which is what lets a
whole turn be exercised with no key and no network. The turn itself — the call sequence, the
budgets, the tool executor, the repair pass — lives in `src/agent/`.

The idea that shapes the rest: **the model never receives a real profile value and never emits
one.** It works with keys (`profile.email`), and the guard substitutes the real value after
validation (P3). That single constraint kills the invented-value failure at its root rather
than catching it downstream.

The provider is OpenAI, with strict structured outputs. `project-overview.md` named Gemini
until P4; the one-module rule it states is what matters and holds either way.
"""
