"""Demo-mode errors."""


class DemoModeDisabled(RuntimeError):
    """Something asked for the demo account while `DEMO_MODE` was off."""

    def __init__(self) -> None:
        super().__init__(
            "DEMO_MODE is not enabled. Set DEMO_MODE=true in .env to use the demo "
            "account. It cannot be enabled when ENV=production.",
        )


class DemoTokenConflict(RuntimeError):
    """`DEMO_TOKEN` hashes to a token that belongs to a different account."""

    def __init__(self) -> None:
        super().__init__(
            "DEMO_TOKEN matches a token owned by another account. Generate a new one "
            "for DEMO_TOKEN; the existing token was left untouched.",
        )
