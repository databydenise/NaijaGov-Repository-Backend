"""Auth constants."""

# The cookie the web app carries. httpOnly, so the browser never holds a readable
# credential and an XSS cannot read it out of storage.
SESSION_COOKIE_NAME = "ngv_session"

JWT_ALGORITHM = "HS256"

# Present in every session JWT so one can never be mistaken for a different kind of token
# added later. `decode_session_jwt` rejects anything else.
JWT_TYPE_SESSION = "session"

# argon2 hashes the whole input, so an unbounded password is a free CPU-burn vector.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128

LOGIN_RATE_LIMIT = 10
LOGIN_RATE_WINDOW_SECONDS = 15 * 60
