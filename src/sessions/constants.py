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

# What an action-log row's `status` may say. The first three are what `/plan` writes when the
# guard refuses something; `changed` and `cancelled` arrive with `/results`, which is the first
# code that knows what actually happened on the page:
#
# - `ok`        — the action ran and the page kept what was written
# - `changed`   — it ran and the portal rewrote the value (spaces stripped from a phone number,
#                 a date reformatted). Accepted-but-changed is not a failure, and collapsing the
#                 two would hide the most common thing a government portal does to input.
# - `failed`    — it ran and did not take
# - `rejected`  — it never ran: refused by the backend guard at plan time, or by the content
#                 script's own validation at write time
# - `cancelled` — it never ran because the batch stopped first (a checkpoint, a navigation, the
#                 page changing underneath, the batch timeout)
#
# Widening this list is a migration (the `ck_action_log_status` CHECK) and a two-repo change.
ACTION_STATUSES: tuple[str, ...] = ("ok", "changed", "failed", "rejected", "cancelled")

# An unbounded JSONB column grows until a prompt gets expensive.
MAX_HISTORY_TURNS = 10
