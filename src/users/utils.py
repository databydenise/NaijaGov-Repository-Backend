"""User helpers with no business logic."""


def normalize_email(raw: str) -> str:
    """
    Trim and lowercase an email before it is stored or looked up.

    The column is `citext`, so this is belt and braces — but it also means what we store is
    what we would print, and that a lookup and an insert agree without relying on the
    database's collation.
    """
    return raw.strip().lower()
