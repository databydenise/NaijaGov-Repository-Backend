"""Workflow constants."""

# How long the registry reads are reused. It changes only when the seed runs, and
# `/context` would otherwise pay for a workflow lookup on every page load.
REGISTRY_CACHE_TTL_SECONDS = 300

ACTIVE_WORKFLOWS_CACHE_KEY = "workflows:active"
STEPS_CACHE_KEY_PREFIX = "workflows:steps:"
