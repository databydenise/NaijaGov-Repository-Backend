"""Profile constants."""

# The fillable fields, in the order the web app shows them and `completeness.missing`
# lists them.
PROFILE_FIELDS: tuple[str, ...] = (
    "full_name",
    "email",
    "phone",
    "address",
    "state",
    "lga",
)

# The only columns a caller may patch. `version` and the timestamps are maintained here
# and by the trigger, never supplied by a request body.
EDITABLE_FIELDS: frozenset[str] = frozenset(PROFILE_FIELDS)
