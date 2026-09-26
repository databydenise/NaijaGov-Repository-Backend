"""Session and action-log constants.

`ACTION_TYPES` mirrors the action union in `naijagov-extension/src/shared/actions.ts`.
It is a CHECK constraint rather than a Postgres enum because the list will change during
the build, and altering a CHECK is a one-line migration while altering an enum is not.
Changing this list is a two-repo change.
"""

ACTION_TYPES: tuple[str, ...] = (
    "fill",
    "select",
    "check",
    "highlight",
    "scroll",
    "explain",
    "clickSafe",
    "pause",
)

ACTION_STATUSES: tuple[str, ...] = ("ok", "failed", "rejected")

# An unbounded JSONB column grows until a prompt gets expensive.
MAX_HISTORY_TURNS = 10
