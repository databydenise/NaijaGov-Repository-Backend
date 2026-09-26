"""Profile reads and writes. No route writes SQL; it calls these."""

import uuid
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.profiles.constants import EDITABLE_FIELDS, PROFILE_FIELDS
from src.profiles.exceptions import UnknownProfileField
from src.profiles.models import Profile
from src.profiles.schemas import Completeness, ProfileOut, ProfileResponse


async def get_profile(db: AsyncSession, user_id: uuid.UUID) -> Profile | None:
    """The user's profile row, or None if they have not saved one yet."""
    result = await db.execute(select(Profile).where(Profile.user_id == user_id))

    return result.scalar_one_or_none()


async def ensure_profile(db: AsyncSession, user_id: uuid.UUID) -> Profile:
    """
    The user's profile, created empty if it does not exist yet.

    Called from the auth dependency so no downstream code has to handle a missing row. It
    must not bump `version`: creating the row a user was always going to have is not an
    edit, and the extension uses `version` to decide whether its cached profile is stale.
    """
    statement = (
        insert(Profile)
        .values(user_id=user_id)
        .on_conflict_do_nothing(index_elements=[Profile.user_id])
        .returning(Profile)
    )

    result = await db.execute(statement)
    profile = result.scalar_one_or_none()

    if profile is not None:
        return profile

    # Nothing was returned, so the row already existed.
    existing = await get_profile(db, user_id)

    if existing is None:  # pragma: no cover - only reachable if the row vanished mid-call
        message = f"Profile for {user_id} could neither be created nor read"
        raise RuntimeError(message)

    return existing


async def load_profile(db: AsyncSession, user_id: uuid.UUID) -> Profile:
    """
    The user's profile for a read endpoint, with no query when it is already loaded.

    `require_user` has run `ensure_profile` in this same session, so `db.get` answers from
    the identity map. `require_token` has not, so there it costs one primary-key SELECT,
    and falls back to creating the row only for an account that somehow lacks one.
    """
    profile = await db.get(Profile, user_id)

    if profile is not None:
        return profile

    return await ensure_profile(db, user_id)


def to_profile_out(profile: Profile) -> ProfileOut:
    """The row as the API shows it. Only the fillable fields, never the ids."""
    return ProfileOut.model_validate(
        {field: getattr(profile, field) for field in PROFILE_FIELDS},
    )


def compute_completeness(profile: ProfileOut) -> Completeness:
    """
    Which fields are filled. A whitespace-only value counts as missing.

    Counted, not stored, so it can never disagree with the row it describes.
    """
    values = profile.model_dump()
    missing = [
        field for field in PROFILE_FIELDS if not (values[field] or "").strip()
    ]

    return Completeness(
        filled=len(PROFILE_FIELDS) - len(missing),
        total=len(PROFILE_FIELDS),
        missing=missing,
    )


async def read_profile(db: AsyncSession, user_id: uuid.UUID) -> ProfileResponse:
    """`GET /profile`: the profile, its version, and how complete it is."""
    profile = await load_profile(db, user_id)
    profile_out = to_profile_out(profile)

    return ProfileResponse(
        profile=profile_out,
        version=profile.version,
        updated_at=profile.updated_at,
        completeness=compute_completeness(profile_out),
    )


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
