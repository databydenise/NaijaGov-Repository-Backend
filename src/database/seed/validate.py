"""
Checks that run across the seed files, before a single row is written.

Pydantic validates each file on its own; these are the checks that need to see the whole
set. They run here rather than relying on the foreign keys so a broken seed names the
offending id instead of failing with an integrity error halfway through a write.
"""

import re

from src.database.seed.exceptions import SeedValidationError
from src.database.seed.schemas import RuleSeed, WorkflowSeed, WorkflowStepSeed

FIRST_STEP_INDEX = 1


def _check_patterns_compile(workflows: list[WorkflowSeed]) -> None:
    """
    Every `url_patterns` entry must compile.

    A pattern that does not compile would raise in the matcher on a live page instead of
    here, where the file that holds it can be named.
    """
    for workflow in workflows:
        for pattern in workflow.url_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                message = (
                    f"Workflow {workflow.id} has a url_pattern that is not a valid "
                    f"regex ({exc}): {pattern}"
                )
                raise SeedValidationError(message) from exc


def _check_step_indexes(steps: list[WorkflowStepSeed]) -> None:
    """
    Each workflow's step indexes run 1, 2, 3 … with no gaps and no repeats.

    A gap means a step was deleted and the rest were never renumbered, and a repeat makes
    "the next step" ambiguous — both are silent in the data and loud on a real page.
    """
    by_workflow: dict[str, list[int]] = {}

    for step in steps:
        by_workflow.setdefault(step.workflow_id, []).append(step.index)

    for workflow_id, indexes in by_workflow.items():
        expected = list(range(FIRST_STEP_INDEX, FIRST_STEP_INDEX + len(indexes)))

        if sorted(indexes) != expected:
            message = (
                f"Workflow {workflow_id} step indexes must run {expected[0]}"
                f"–{expected[-1]} with no gaps or repeats, got {sorted(indexes)}"
            )
            raise SeedValidationError(message)


def _check_step_references(
    steps: list[WorkflowStepSeed],
    workflow_ids: set[str],
) -> None:
    """Every step belongs to a workflow in the file, and carries its id as a prefix."""
    for step in steps:
        if step.workflow_id not in workflow_ids:
            message = f"Step {step.id} references unknown workflow {step.workflow_id}"
            raise SeedValidationError(message)

        if not step.id.startswith(f"{step.workflow_id}."):
            message = f"Step id {step.id} must be prefixed with its workflow id"
            raise SeedValidationError(message)


def _check_rule_references(
    rules: list[RuleSeed],
    workflow_ids: set[str],
    steps_by_id: dict[str, WorkflowStepSeed],
) -> None:
    """
    Every rule points at a workflow, and at a step of that same workflow when it names one.

    The last check matters: a rule attached to another workflow's step would never be
    retrieved, because retrieval filters on workflow and step together.
    """
    for rule in rules:
        if rule.workflow_id not in workflow_ids:
            message = f"Rule {rule.id} references unknown workflow {rule.workflow_id}"
            raise SeedValidationError(message)

        if rule.step_id is None:
            continue

        step = steps_by_id.get(rule.step_id)

        if step is None:
            message = f"Rule {rule.id} references unknown step {rule.step_id}"
            raise SeedValidationError(message)

        if step.workflow_id != rule.workflow_id:
            message = (
                f"Rule {rule.id} is on workflow {rule.workflow_id} but its step "
                f"{rule.step_id} belongs to {step.workflow_id}"
            )
            raise SeedValidationError(message)


def _check_unique_ids(
    workflows: list[WorkflowSeed],
    steps: list[WorkflowStepSeed],
    rules: list[RuleSeed],
) -> None:
    """
    No id appears twice in a file.

    An upsert would quietly keep whichever row was written last, so a duplicate would make
    the seed's result depend on file order.
    """
    for label, ids in (
        ("workflow", [workflow.id for workflow in workflows]),
        ("step", [step.id for step in steps]),
        ("rule", [rule.id for rule in rules]),
    ):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})

        if duplicates:
            message = f"Duplicate {label} ids: {', '.join(duplicates)}"
            raise SeedValidationError(message)


def check_seed_data(
    workflows: list[WorkflowSeed],
    steps: list[WorkflowStepSeed],
    rules: list[RuleSeed],
) -> None:
    """Run every cross-file check. Raises `SeedValidationError` on the first failure."""
    workflow_ids = {workflow.id for workflow in workflows}
    steps_by_id = {step.id: step for step in steps}

    _check_unique_ids(workflows, steps, rules)
    _check_patterns_compile(workflows)
    _check_step_references(steps, workflow_ids)
    _check_step_indexes(steps)
    _check_rule_references(rules, workflow_ids, steps_by_id)
