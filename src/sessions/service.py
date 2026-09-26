"""Session reads, writes, and expiry.

Every read filters on `expires_at > now()`, so an expired row can never be handed back
even if the cleanup CLI has not run for a week.
"""

import uuid
from datetime import timedelta

from sqlalchemy import case, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.sessions.constants import MAX_HISTORY_TURNS
from src.sessions.models import Session


async def get_or_create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    workflow_id: str | None = None,
    step_id: str | None = None,
    page_hash: str | None = None,
) -> Session:
    """The user's live session, created if there isn't one.

    Reuses the most recently touched unexpired session for this user — scoped to
    `workflow_id` when one is given, so moving to a different portal starts a new session
    rather than inheriting another portal's history. Any of `workflow_id`, `step_id`, and
    `page_hash` that are passed overwrite what the row held; None leaves the column alone.
    """
    statement = (
        select(Session)
        .where(
            Session.user_id == user_id,
            Session.expires_at > func.now(),
        )
        .order_by(Session.updated_at.desc())
        .limit(1)
    )

    if workflow_id is not None:
        statement = statement.where(Session.workflow_id == workflow_id)

    result = await db.execute(statement)
    session = result.scalar_one_or_none()

    if session is None:
        session = Session(
            user_id=user_id,
            workflow_id=workflow_id,
            step_id=step_id,
            page_hash=page_hash,
            expires_at=func.now() + timedelta(hours=settings.session_ttl_hours),
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)

        return session

    if workflow_id is not None:
        session.workflow_id = workflow_id
    if step_id is not None:
        session.step_id = step_id
    if page_hash is not None:
        session.page_hash = page_hash

    await db.flush()

    return session


async def get_session_for_tab(
    db: AsyncSession,
    user_id: uuid.UUID,
    tab_id: int,
) -> Session | None:
    """
    The live session for this user's tab, or None.

    Filters the expiry here rather than in the caller, so an expired row can never be
    handed back as a cache hit even if the cleanup CLI has not run.
    """
    statement = select(Session).where(
        Session.user_id == user_id,
        Session.tab_id == tab_id,
        Session.expires_at > func.now(),
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def upsert_tab_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    tab_id: int,
    workflow_id: str | None,
    step_id: str | None,
    page_hash: str,
    cached_rule_ids: list[str],
) -> Session:
    """
    Write this tab's session, replacing whatever it held.

    One statement, conflicting on `(user_id, tab_id)`: `/context` is called on every DOM
    mutation, so two requests racing on one tab is ordinary rather than exceptional, and a
    read-then-write would leave two rows.

    `history` is cleared when the row being replaced had already expired. A browser reuses
    tab ids after a restart, so the row found under this id can belong to a tab that no
    longer exists — carrying its chat forward would show one page's conversation on
    another, after the point where it was meant to have been forgotten.
    """
    expires_at = func.now() + timedelta(hours=settings.session_ttl_hours)

    statement = (
        insert(Session)
        .values(
            user_id=user_id,
            tab_id=tab_id,
            workflow_id=workflow_id,
            step_id=step_id,
            page_hash=page_hash,
            cached_rule_ids=cached_rule_ids,
            expires_at=expires_at,
        )
        .on_conflict_do_update(
            index_elements=[Session.user_id, Session.tab_id],
            set_={
                "workflow_id": workflow_id,
                "step_id": step_id,
                "page_hash": page_hash,
                "cached_rule_ids": cached_rule_ids,
                "expires_at": expires_at,
                "history": case(
                    (Session.expires_at <= func.now(), text("'[]'::jsonb")),
                    else_=Session.history,
                ),
            },
        )
        .returning(Session)
    )

    # populate_existing, because a Session already in this session's identity map would
    # otherwise come back with its pre-write values.
    result = await db.execute(statement, execution_options={"populate_existing": True})

    return result.scalar_one()


def trim_history(history: list[dict[str, object]]) -> list[dict[str, object]]:
    """The last `MAX_HISTORY_TURNS` turns. Applied before any write to `history`."""
    return history[-MAX_HISTORY_TURNS:]


async def delete_expired_sessions(db: AsyncSession) -> int:
    """Delete every session past its expiry. Returns how many rows went.

    `action_log` rows cascade with their session, which is the point: the log lives exactly
    as long as the session it describes.
    """
    result = await db.execute(delete(Session).where(Session.expires_at <= func.now()))

    return result.rowcount
