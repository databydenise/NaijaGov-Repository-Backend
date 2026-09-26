"""
How a rule leaves this service.

No endpoint returns one yet — `/explain` is B9 — but the shape is fixed here so that when
one does, it cannot be shipped without its provenance. Every field below is part of the
answer, not decoration: a requirement shown without its source, its date, and whether it
is demo content is exactly the failure this project is built to avoid.
"""

from datetime import date

from pydantic import BaseModel


class RuleOut(BaseModel):
    """One rule, with the provenance that must travel with it."""

    id: str
    topic: str
    field_label: str | None
    requirement: str

    # Shown with every answer that uses this rule. Never omitted to save space.
    source_url: str
    last_checked: date

    # True means invented demo content. The panel shows it in place of the source line, so
    # a plausible-looking requirement is never mistaken for official guidance.
    is_placeholder: bool
