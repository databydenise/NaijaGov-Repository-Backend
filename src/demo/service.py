"""
The demo account.

A judge should be able to use the product without signing up. The account is ordinary: same
tables, same hashing, same token rules, no special powers. The only difference is that its
password is known and its token is fixed in `.env`.

Set up by the server itself at startup when `DEMO_MODE=true`, so `fastapi dev` is all anyone
runs. Every step is idempotent, because `fastapi dev` restarts on each file save: running it
a hundred times leaves the same account, the same profile, and the same working token.

Guards: this refuses to run unless `DEMO_MODE=true`, and `src/config.py` refuses to boot at
all when `DEMO_MODE=true` with `ENV=production`.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.security import hash_password, verify_password
from src.config import settings
from src.demo.exceptions import DemoModeDisabled, DemoTokenConflict
from src.profiles import service as profiles_service
from src.tokens import service as tokens_service
from src.tokens.utils import LAST4_LENGTH, hash_token
from src.users import service as users_service
from src.users.models import User

logger = logging.getLogger(__name__)

DEMO_TOKEN_TTL_DAYS = 365
DEMO_TOKEN_LABEL = "Demo device"  # noqa: S105 - this is a display label, not a credential

# Obviously fictional. No real person's details, no realistic-looking ID numbers, and
# nothing that could be mistaken for a genuine record.
DEMO_PROFILE = {
    "full_name": "Demo User",
    "email": "demo@example.com",
    # Eleven digits starting with zero, because that is what the seeded rule for the
    # Phone Number field requires. A demo profile that fails its own workflow's rule makes
    # the fill preview contradict the explanation shown beside it.
    "phone": "08000000000",
    "address": "000 Example Close, Demo District",
    "state": "Example State",
    "lga": "Example LGA",
}


@dataclass(frozen=True)
class DemoAccountStatus:
    """What startup did, for one log line. Carries no credential."""

    user_created: bool
    profile_filled: bool
    token_installed: bool
    token_configured: bool


async def _ensure_user(db: AsyncSession) -> tuple[User, bool]:
    """The demo user, created if missing and re-keyed if `DEMO_PASSWORD` has changed."""
    user = await users_service.get_user_by_email(db, settings.demo_email)

    if user is None:
        try:
            created = await users_service.create_user(
                db,
                settings.demo_email,
                hash_password(settings.demo_password),
            )
        except IntegrityError:
            # Another worker created it between our lookup and our insert.
            await db.rollback()
            existing = await users_service.get_user_by_email(db, settings.demo_email)

            if existing is None:  # pragma: no cover - the conflicting row vanished
                raise

            return existing, False

        return created, True

    # One argon2 verification per startup, so editing DEMO_PASSWORD in .env takes effect on
    # the next restart without a manual step.
    if not verify_password(settings.demo_password, user.password_hash):
        await users_service.set_password_hash(
            db,
            user.id,
            hash_password(settings.demo_password),
        )

    return user, False


async def _ensure_profile(db: AsyncSession, user: User) -> bool:
    """
    Fill the demo profile the first time only. Returns whether it filled it.

    Rewriting it on every restart would bump `version` on every file save, and the extension
    reads `version` to decide whether its cached profile is stale. It would also undo any
    edit a judge made during the demo.
    """
    profile = await profiles_service.ensure_profile(db, user.id)

    if profile.full_name is not None:
        return False

    await profiles_service.upsert_profile(db, user.id, DEMO_PROFILE)

    return True


async def _ensure_token(db: AsyncSession, user: User, raw: str) -> bool:
    """
    Make the configured `DEMO_TOKEN` the demo user's one active token.

    Returns whether anything changed. A token that is already active is left alone, so its
    `last_used_at` survives restarts. One that was revoked — say, a judge trying the revoke
    button — is restored, which means a restart always puts the demo back in a working state.
    """
    token_hash = hash_token(raw)
    expires_at = datetime.now(UTC) + timedelta(days=DEMO_TOKEN_TTL_DAYS)
    existing = await tokens_service.get_token_row_by_hash_any_state(db, token_hash)

    if existing is not None and existing.user_id != user.id:
        raise DemoTokenConflict

    is_active = (
        existing is not None
        and existing.revoked_at is None
        and existing.expires_at > datetime.now(UTC)
    )

    if is_active:
        return False

    # Clears any token the demo user issued through the web app, and satisfies the
    # one-active-token index before the demo token is (re)activated.
    await tokens_service.revoke_active_tokens(db, user.id)

    if existing is not None:
        await tokens_service.reactivate_token(db, existing.id, expires_at)

        return True

    await tokens_service.create_token(
        db,
        user_id=user.id,
        token_hash=token_hash,
        last4=raw[-LAST4_LENGTH:],
        label=DEMO_TOKEN_LABEL,
        expires_at=expires_at,
    )

    return True


async def ensure_demo_account(db: AsyncSession) -> DemoAccountStatus:
    """Create or repair the demo account. Safe to call on every startup."""
    if not settings.demo_mode:
        raise DemoModeDisabled

    user, user_created = await _ensure_user(db)
    profile_filled = await _ensure_profile(db, user)

    token_installed = False
    if settings.demo_token:
        token_installed = await _ensure_token(db, user, settings.demo_token)

    return DemoAccountStatus(
        user_created=user_created,
        profile_filled=profile_filled,
        token_installed=token_installed,
        token_configured=settings.demo_token is not None,
    )
