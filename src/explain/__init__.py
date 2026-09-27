"""
`POST /explain` — one field, in plain English, with a source.

The endpoint the demo leans on hardest, because it works on any page whether or not anything can
be filled. Most of the code here is about **not** calling a model: a sensitive field answers from
fixed copy, a retrieval miss answers with the no-guidance sentence, and a repeat click answers
from a table. Only a genuine hit on the corpus reaches `agent/explain.py`.

Nothing here writes to the session. `/explain` is a side call from READY, not a step in a
workflow: no history entry, no chat values, no plan. The one write it makes is to its own cache,
which is shared across users and holds no user's data.
"""
