"""Profile reads and writes. No route writes SQL; it calls these."""

import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.profiles.constants import EDITABLE_FIELDS
from src.profiles.exceptions import UnknownProfileField
from src.profiles.models import Profile


async def get_profile(db: AsyncSession, user_id: uuid.UUID) -> Profile | None:
    """The user's profile row, or None if they have not saved one yet."""
    result = await db.execute(select(Profile).where(Profile.user_id == user_id))

    return result.scalar_one_or_none()


async def upsert_profile(
    db: AsyncSession,
    user_id: uuid.UUID,
    patch: Mapping[str, str | None],
) -> Profile:
    """Create or update a profile, bumping `version` by exactly one per write.

    The bump happens in SQL (`version = profiles.version + 1`) rather than by reading the
    row first, so two concurrent writes cannot both land on the same version number.
    """
    unknown = set(patch) - EDITABLE_FIELDS
    if unknown:
        raise UnknownProfileField(unknown)

    values = dict(patch)

    statement = (
        insert(Profile)
        .values(user_id=user_id, **values)
        .on_conflict_do_update(
            index_elements=[Profile.user_id],
            set_={**values, "version": Profile.version + 1},
        )
        .returning(Profile)
    )

    # Without populate_existing, a Profile already loaded in this session would come back
    # from the identity map with its pre-write values — including the old version number.
    result = await db.execute(
        statement,
        execution_options={"populate_existing": True},
    )

    return result.scalar_one()
