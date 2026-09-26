"""Profile-specific errors."""


class UnknownProfileField(ValueError):
    """A patch named a column that is not user-editable."""

    def __init__(self, fields: set[str]) -> None:
        self.fields = fields

        super().__init__(f"Not editable profile fields: {sorted(fields)}")
