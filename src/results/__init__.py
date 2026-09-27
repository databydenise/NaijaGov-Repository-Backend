"""
`POST /results` — what actually happened on the page, in statuses and ids.

The smallest endpoint in the system and the one that makes the rest measurable: which fields fill
reliably, which portals reject a programmatic write, how often a checkpoint fires. It also decides
the one sentence the panel shows at the end of a run, from a fixed table rather than a model call.

It receives statuses and ids. No values, ever — what went into the form is already where it
belongs, in the user's browser.
"""
