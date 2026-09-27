"""
`/explain` request and response models.

The request carries **one field**, not a snapshot, and it reuses `context/schemas.py::PageField`
wholesale rather than restating its six attributes. That reuse is not tidiness: `PageField` carries
the tripwire that refuses a field object containing a `value`, and a second definition of the same
shape is how one of the two quietly loses it. The one thing that must never arrive here is what a
citizen typed into the field they are asking about.

The response carries no values either — an explanation, an example the model made up from the
label, and the sources behind any claim it makes. `/explain` is the one endpoint on this service
that cannot cause anything to be written to a page, and its response shape is why: there is no
action list to put one in.
"""

import uuid
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field

from src.ai.context import StepView
from src.context.schemas import PageField, SnapshotModel
from src.explain.constants import MAX_NEARBY_TEXT_CHARS, MAX_QUESTION_CHARS


class ExplainRequest(SnapshotModel):
    """
    What the panel sends when the user clicks Explain on a field.

    `session_id` is optional, and a wrong one is not an error. It is read for context — which
    workflow, which step, whose agency's material to search — and nothing here is about to be
    written, so a session that has expired, belongs to another account, or was never opened
    degrades to an answer with no agency filter. `/plan` refuses in the same situation because
    `/plan` is about to fill a form; this endpoint is about to say a sentence.
    """

    session_id: uuid.UUID | None = None

    # Nested rather than flattened, so the `value` tripwire, the 200-character label cap and the
    # option cap all apply exactly as they do on a snapshot.
    field: PageField

    # The portal's own instruction beside the field. Optional, capped, and treated as untrusted
    # page text: it is fenced and escaped before the model sees it.
    nearby_text: Annotated[str, Field(max_length=MAX_NEARBY_TEXT_CHARS)] = ""

    # The user's own question about the field, where they had one. An answer to a custom question
    # is never cached — see `store.py` — so this text never reaches a shared table.
    question: Annotated[str, Field(max_length=MAX_QUESTION_CHARS)] | None = None


class SourceOut(BaseModel):
    """
    One source behind the explanation, as the panel renders it.

    `checked` is when the corpus last read that source, not when this answer was written. A source
    whose date we do not hold shows none rather than borrowing today's — the panel prints
    "Source: FRSC FAQ · checked 12 Sep 2026", and a date we invented there would undermine the
    only line on the card that makes the rest of it trustworthy.
    """

    title: str
    url: str
    checked: str | None = None


class ExplainResponse(BaseModel):
    """
    What the panel needs to render the explain card.

    `grounded` is false for exactly two answers: the corpus does not cover the field, and the
    lookup could not be made. Both come with an empty `sources` and a sentence that says so, so the
    panel can show the card without a source line rather than showing a claim without a source.

    A sensitive field's answer is `grounded: true` with no sources, which is not a contradiction:
    "type this yourself, I never handle codes" is a fact about this product, not a claim about the
    government process, and there is no official document to cite for it.
    """

    field_id: str
    explanation: str
    example: str | None = None
    sources: list[SourceOut] = Field(default_factory=list)
    grounded: bool
    cached: bool = False


@dataclass(frozen=True)
class ExplainScope:
    """
    Where the question was asked, as far as the registry knows.

    A frozen dataclass rather than a pydantic model, like `documents/schemas.py`: it crosses no
    HTTP boundary and parses nothing untrusted — it is read out of the session and the registry,
    both of which are ours.

    Every field is optional because every field can legitimately be absent. A page outside the
    registry, an expired session, another account's session id: all of them produce a scope that
    claims nothing, which means no agency filter on the search and no step line in the prompt.
    That is a worse answer, not an error — and the alternative, a step line invented for a
    workflow we have never seen, is a wrong answer dressed as a good one.

    `service` is the workflow's own name. The `workflows` table has no `service` column, and the
    workflow's name ("Business Name Registration") is the closest thing it holds to what the corpus
    calls a service ("Driver's Licence") — a specific offering under an agency, which is what the
    retrieval filter is for.
    """

    workflow_id: str | None = None
    step_id: str | None = None
    agency: str | None = None
    service: str | None = None

    # For the prompt's `STEP:` line. Absent for a page with no workflow.
    step: StepView | None = None

