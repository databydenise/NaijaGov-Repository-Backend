"""Profile constants."""

# The only columns a caller may patch. `version` and the timestamps are maintained here
# and by the trigger, never supplied by a request body.
EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "full_name",
        "email",
        "phone",
        "address",
        "state",
        "lga",
    },
)
