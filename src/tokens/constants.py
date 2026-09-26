"""Token constants."""

# Issuing is cheap but not free, and a loop that regenerates forever would fill the table.
ISSUE_RATE_LIMIT = 10
ISSUE_RATE_WINDOW_SECONDS = 60 * 60
