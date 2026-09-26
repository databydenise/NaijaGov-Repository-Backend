"""Imports every ORM model so `Base.metadata` is complete.

Alembic's autogenerate and the seed runner both need every table registered. A model that
is not imported here is invisible to both, which shows up as a migration that silently
omits a table.
"""

from src.database.base import Base
from src.knowledge.models import Rule
from src.profiles.models import Profile
from src.sessions.models import ActionLog, Session
from src.tokens.models import ExtensionToken
from src.users.models import User
from src.workflows.models import Workflow, WorkflowStep

__all__ = [
    "ActionLog",
    "Base",
    "ExtensionToken",
    "Profile",
    "Rule",
    "Session",
    "User",
    "Workflow",
    "WorkflowStep",
]
