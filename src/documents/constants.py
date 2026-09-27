"""
Retrieval constants, in one place.

The distance cutoff is the feature. Everything else in this package exists so that
"I have no official guidance on that" is a reachable outcome, and this number is what
decides when it is reached.
"""

from typing import Final

# --- The cutoff ---------------------------------------------------------------------
#
# Cosine distance above which a chunk is treated as no match at all. 0.45 is the value the
# spec proposes and is what ships until a sweep over a real corpus replaces it:
# `python -m scripts.tune_threshold` prints precision and recall per candidate cutoff, and
# the value to pick is the strictest one where no unanswerable query returns a chunk.
#
# NOT YET MEASURED. The corpus has never been ingested from this machine, so this number is
# inherited, not chosen. Slightly too strict is the right way to be wrong: a missing answer
# is a shrug, and a confident wrong fee is the bug the prototype already produced.
#
# Override per deployment with RETRIEVAL_MAX_DISTANCE.
DEFAULT_RETRIEVAL_MAX_DISTANCE: Final = 0.45

# Candidate cutoffs the tuning sweep walks, low (strict) to high (loose).
TUNING_CUTOFFS: Final = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

# --- Query shaping ------------------------------------------------------------------

# How many chunks a caller may ask for. More than five is more context than an answer
# needs and more text for the guard to check; fewer than one is not a search.
MIN_SEARCH_LIMIT: Final = 1
MAX_SEARCH_LIMIT: Final = 5
DEFAULT_SEARCH_LIMIT: Final = 3

# A query longer than this is a page paste, not a question. Truncated rather than refused:
# the opening of a long question still carries what it is asking.
MAX_QUERY_CHARS: Final = 1000

# --- Budgets ------------------------------------------------------------------------

# Postgres-side statement timeout for the vector search. A slow scan must not hold up a
# `/plan` the user is watching; the search gives up and the caller reports unavailable.
SEARCH_STATEMENT_TIMEOUT_MS: Final = 2000

# How long the embedding call may take before the search is abandoned. Together with the
# statement timeout this bounds a search at roughly five seconds in the worst case, well
# inside the point where a user assumes the panel has hung.
EMBEDDING_TIMEOUT_SECONDS: Final = 3.0

# --- Query embedding cache ----------------------------------------------------------

# Repeat questions are common — the same field, the same fee, asked on every page load —
# and each embedding is a round trip. Keyed by model plus a hash of the query text, so a
# model change cannot serve a stale vector and no raw question text sits in a global dict.
QUERY_EMBEDDING_CACHE_TTL_SECONDS: Final = 300
QUERY_EMBEDDING_CACHE_PREFIX: Final = "documents:query_embedding:"

# Bounds what a flood of distinct queries can cost in memory. Above this the cache evicts
# rather than grows.
QUERY_EMBEDDING_CACHE_MAX_ENTRIES: Final = 512

# --- Corpus statistics for /health --------------------------------------------------

CORPUS_STATS_CACHE_KEY: Final = "documents:corpus_stats"
CORPUS_STATS_CACHE_TTL_SECONDS: Final = 300

# --- Embeddings ---------------------------------------------------------------------

# text-embedding-3-small. The column is `vector(1536)`, so a model of another width cannot
# be swapped in without a migration, which is the point: a silent width change would
# corrupt every comparison.
EMBEDDING_DIMENSIONS: Final = 1536


class RetrievalStatus:
    """What `/health` reports for retrieval."""

    # Key present, database answered.
    OK: Final = "ok"

    # No OPENAI_API_KEY. The app runs, searches return nothing and say so.
    UNCONFIGURED: Final = "unconfigured"

    # Configured, but the corpus could not be counted. Liveness is unaffected.
    UNAVAILABLE: Final = "unavailable"


class SourceKind:
    """The three shapes an ingested source comes in. Stored as `documents.source_kind`."""

    FAQ: Final = "faq"
    PAGE: Final = "page"
    PDF: Final = "pdf"

    ALL: Final = (FAQ, PAGE, PDF)
