"""
The registry as the rest of the service reads it.

Dataclasses rather than Pydantic models: these are cached in memory and never cross the
API boundary, and one of them holds a compiled `re.Pattern`, which Pydantic would need to
be talked into accepting. Frozen, because a cached value handed to a request must not be
something that request can edit.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveWorkflow:
    """A workflow with its URL patterns already compiled.

    Compiling here rather than per request matters on `/context`, which has a 300 ms budget
    and would otherwise recompile every pattern on every page load.
    """

    id: str
    agency: str
    name: str
    url_patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class Step:
    """One step of a workflow, in the order the portal presents it."""

    id: str
    workflow_id: str
    name: str
    index: int

    # Raw page labels. B8 normalises them when matching; it does not change them here.
    field_labels: tuple[str, ...]

    # What lets `/context` warn that the next button submits the application.
    is_final: bool
