"""
The runner's internal abort.

`TurnAborted` is raised deep in the sequence — a budget check, a dead provider, a held lock —
and caught once, at the entry point, where it becomes a `TurnFailure` with a sentence for the
user. It never leaves this package.

Raising rather than threading a union through every helper is deliberate: the sequence in
`service.py` reads as the nine steps the spec describes, with guard clauses, instead of six
functions each returning "a result or a reason".
"""

from src.agent.constants import FAILURE_MESSAGES, FailureCode


class TurnAborted(Exception):
    """A turn that cannot continue, with the code and the user's sentence already chosen."""

    def __init__(self, code: FailureCode) -> None:
        self.code: FailureCode = code
        self.message: str = FAILURE_MESSAGES[code]
        super().__init__(code)
