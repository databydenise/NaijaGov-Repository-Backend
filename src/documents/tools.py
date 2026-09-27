"""
The search tool, as the model sees it.

P4 passes this to the model and executes `service.search_government_information` when the
model calls it. Nothing here calls a model or a database; it is the schema and the result
shape, kept beside the function it describes so the two cannot drift.

The description is load-bearing. A tool that can return nothing is only safe if the model
has been told, in the tool itself, what nothing means — otherwise an empty result reads as
"the search broke, answer it yourself", which is the failure this whole package prevents.
"""

from typing import Any, Final

from src.documents.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT,
)
from src.documents.schemas import RetrievalResult, RetrievedChunk

SEARCH_TOOL_NAME: Final = "search_government_information"

# Said twice on purpose — once in the tool description and once in the result payload —
# because a model that has filled its context with page text reads the nearest instruction.
EMPTY_RESULT_INSTRUCTION: Final = (
    "Returns an empty list when no official source covers the question. If it does, say "
    "you do not have official guidance on that and suggest where the user might check — "
    "do not answer from your own knowledge."
)

SEARCH_TOOL_DESCRIPTION: Final = (
    "Search the stored corpus of official Nigerian government source material — agency "
    "FAQs, portal guidance, and published documents — for what it says about a question. "
    "Use it for any requirement, fee, timeline, or document a user asks about. "
    + EMPTY_RESULT_INSTRUCTION
)

# Plain JSON Schema, kept separate from any one provider's envelope. The two SDKs this
# project may use wrap it differently but agree on the parameter schema itself.
SEARCH_TOOL_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The question to look up, in the user's own words. A full question "
                "retrieves better than keywords."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "How many source chunks to retrieve.",
            "default": DEFAULT_SEARCH_LIMIT,
            "minimum": MIN_SEARCH_LIMIT,
            "maximum": MAX_SEARCH_LIMIT,
        },
        "agency": {
            "type": "string",
            "description": (
                "Restrict to one agency, exactly as the workflow names it, e.g. "
                "'Federal Road Safety Corps (FRSC)'. Omit to search everything."
            ),
        },
        "service": {
            "type": "string",
            "description": (
                'Restrict to one service, e.g. "Driver\'s Licence". Omit to search '
                "everything."
            ),
        },
    },
    "required": ["query"],
}

# OpenAI's function-tool envelope, which is what the prototype used and what P4 starts
# from. A Gemini `FunctionDeclaration` takes the same name, description and parameters.
SEARCH_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": SEARCH_TOOL_DESCRIPTION,
        "parameters": SEARCH_TOOL_PARAMETERS,
    },
}


def chunk_to_tool_item(chunk: RetrievedChunk) -> dict[str, Any]:
    """
    One chunk as the model sees it.

    `chunk_id` is not decoration: the guard in P3 checks every citation against the ids
    that were actually retrieved, and a citation it cannot match is dropped. Distance is
    included so the model can tell a close match from a marginal one.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "content": chunk.content,
        "source_url": chunk.source_url,
        "agency": chunk.agency,
        "service": chunk.service,
        "relevance": round(chunk.score, 4),
    }


def result_to_tool_payload(result: RetrievalResult) -> dict[str, Any]:
    """
    The tool's JSON result.

    The three outcomes are kept apart, because they call for different answers:

    - chunks returned      → ground the answer in them and cite their `chunk_id`s
    - `found: false`       → the corpus has nothing; say so
    - `available: false`   → the search did not run; say the lookup failed, and do not
                             present it as evidence that no rule exists
    """
    if not result.available:
        return {
            "available": False,
            "found": False,
            "chunks": [],
            "note": (
                "The official-information lookup could not be completed. Tell the user "
                "you could not check the official sources right now. Do not answer the "
                "question from your own knowledge."
            ),
        }

    if result.is_empty:
        return {
            "available": True,
            "found": False,
            "chunks": [],
            "note": EMPTY_RESULT_INSTRUCTION,
        }

    return {
        "available": True,
        "found": True,
        "chunks": [chunk_to_tool_item(chunk) for chunk in result.chunks],
    }
