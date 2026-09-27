"""Retrieval over scraped official sources.

`service.py` answers questions with evidence — chunks, sources, distances — and never
with prose. Turning evidence into an answer belongs to the agent, under the guard, so
that the guard can check every claim against text that was actually retrieved.

The one rule that matters here: **an empty result is a valid answer.** A question no
stored source covers must come back with nothing, so the agent says it has no official
guidance rather than inventing a fee.
"""
