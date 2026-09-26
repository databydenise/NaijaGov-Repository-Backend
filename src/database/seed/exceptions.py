"""Seed errors."""


class SeedValidationError(Exception):
    """A seed file is malformed, or references something that is not in the seed set."""
