"""Workflow constants."""

# How long `/me` reuses the supported-host list. It only changes when the seed runs, and
# `/me` is called every time the panel opens.
SUPPORTED_HOSTS_TTL_SECONDS = 300
