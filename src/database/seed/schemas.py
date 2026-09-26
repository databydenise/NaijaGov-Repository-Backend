"""Validation for the seed data files.

Every file is validated before a single row is written. `extra="forbid"` means a typo in a
key fails the seed instead of silently landing as a default, and `source_url` /
`last_checked` are required, so a rule without a source cannot get in.
"""

from datetime import date

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class SeedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkflowSeed(SeedModel):
    id: str = Field(min_length=1)
    agency: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url_patterns: list[str] = Field(default_factory=list)
    is_active: bool = True


class WorkflowStepSeed(SeedModel):
    id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    index: int = Field(ge=0)
    field_labels: list[str] = Field(default_factory=list)
    is_final: bool = False


class RuleSeed(SeedModel):
    id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    step_id: str | None = None
    topic: str = Field(min_length=1)
    field_label: str | None = None
    requirement: str = Field(min_length=1)

    # No default on either. A rule with no source is not a rule.
    source_url: AnyHttpUrl
    last_checked: date
