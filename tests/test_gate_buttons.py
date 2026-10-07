"""Кнопки гейта бюджета. ТЗ §6.3, §4.5, §38 №8."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bot.handlers.gate import (
    ANSWER_CANCEL,
    ANSWER_NOTES,
    ANSWER_STALE,
    CANCEL,
    FULL,
    NOTES_ONLY,
    gate_keyboard,
    handle_cancel,
    handle_full,
    handle_notes_only,
)
from core.db.models import Material, User
from core.db.session import session_scope
from core.ingest.awaiting import is_awaiting, park

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
OWNER_TG = 1100
STRANGER_TG = 1101


async def _parked_material(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], tg_id: int = OWNER_TG
) -> int:
    async with engine.begin() as connection:
        user_id = (
            await connection.execute(
                insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
            )
        ).scalar_one()
        material_id = int(
            (
                await connection.execute(
                    insert(Material)
                    .values(
                        user_id=user_id,
                        kind="text",
                        status="processing",
                        mode="full",
                        completed_stages=[],
                        origin={"path": "/tmp/x.md"},
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )
    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")
    return material_id


def _callback(data: str, tg_id: int = OWNER_TG) -> CallbackQuery:
    callback = CallbackQuery(
        id="cb1",
        from_user=TgUser(id=tg_id, is_bot=False, first_name="T"),
        chat_instance="x",
        data=data,
        message=Message(
            message_id=1,
            date=datetime.datetime(2026, 1, 1),
            chat=Chat(id=tg_id, type="private"),
        ),
    )
    object.__setattr__(callback, "answer", AsyncMock())
    return callback


class FakeArq:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple[object, ...]]] = []

    async def enqueue_job(self, name: str, *args: object) -> None:
        self.jobs.append((name, args))


async def _mode_and_status(engine: AsyncEngine, material_id: int) -> tuple[str, str]:
    async with engine.connect() as connection:
        mode, status = (
            await connection.execute(
                select(Material.mode, Material.status).where(Material.id == material_id)
            )
        ).one()
    return str(mode), str(status)


# --- Клавиатура -----------------------------------------------------------


def test_keyboard_has_three_buttons_from_spec() -> None:
    """§6.3 задаёт ровно три кнопки, и порядок у них оттуда же."""
    rows = gate_keyboard(42).inline_keyboard
    assert len(rows) == 3
    texts = [row[0].text for row in rows]
    assert texts == ["Обработать полностью", "Только конспект, без тестов", "Отмена"]


def test_keyboard_carries_material_id() -> None:
    """`material_id` в кнопке, а не в состоянии FSM.

    Между вопросом и ответом может пройти сутки, за которые пользователь
    успеет прислать другие материалы, и состояние FSM относилось бы к
    последнему из них.
    """
    for row in gate_keyboard(42).inline_keyboard:
        assert row[0].callback_data is not None
        assert row[0].callback_data.endswith(":42")


# --- Нажатия --------------------------------------------------------------


async def test_full_resumes_processing(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """«Обработать полностью»: ожидание снято, задача снова в очереди (§4.5)."""
    material_id = await _parked_material(engine, sessions)
    arq = FakeArq()

    async with session_scope(sessions) as session:
        await handle_full(_callback(f"{FULL}:{material_id}"), session, arq)  # type: ignore[arg-type]
        assert await is_awaiting(session, material_id) is False

    assert arq.jobs == [("process_material_task", (material_id, "/tmp/x.md"))]
    assert await _mode_and_status(engine, material_id) == ("full", "processing")


async def test_notes_only_sets_mode_and_resumes(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """«Только конспект»: режим `notes_only`, обработка продолжается (§6.3)."""
    material_id = await _parked_material(engine, sessions)
    arq = FakeArq()
    callback = _callback(f"{NOTES_ONLY}:{material_id}")

    async with session_scope(sessions) as session:
        await handle_notes_only(callback, session, arq)  # type: ignore[arg-type]

    assert await _mode_and_status(engine, material_id) == ("notes_only", "processing")
    assert len(arq.jobs) == 1
    callback.answer.assert_awaited_once_with(ANSWER_NOTES)  # type: ignore[attr-defined]


async def test_cancel_marks_failed_with_a_reason(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """«Отмена»: `failed` с причиной, и задача в очередь не ставится.

    Причина нужна, потому что `failed` без текста выглядит сбоем системы, хотя
    пользователь сам отменил.
    """
    material_id = await _parked_material(engine, sessions)
    arq = FakeArq()
    callback = _callback(f"{CANCEL}:{material_id}")

    async with session_scope(sessions) as session:
        await handle_cancel(callback, session, arq)  # type: ignore[arg-type]

    async with engine.connect() as connection:
        status, error = (
            await connection.execute(
                select(Material.status, Material.error).where(Material.id == material_id)
            )
        ).one()
    assert status == "failed"
    assert error is not None and "отменена вами" in error
    assert arq.jobs == [], "отмена поставила задачу в очередь"
    callback.answer.assert_awaited_once_with(ANSWER_CANCEL)  # type: ignore[attr-defined]


# --- §38 №8: старый callback не меняет данные -----------------------------


async def test_press_on_material_that_is_not_awaiting_changes_nothing(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§38 №8: старый callback не изменяет данные.

    Кнопка остаётся в чате навсегда, и нажатие через сутки после тайм-аута не
    должно ни возобновлять обработку, ни менять режим. Проверяется и ответ, и
    отсутствие задачи в очереди: ответ без второго ассерта прошёл бы и у
    обработчика, который возобновил обработку и заодно сказал «не актуально».
    """
    material_id = await _parked_material(engine, sessions)
    async with session_scope(sessions) as session:
        from core.ingest.awaiting import resume

        await resume(session, material_id)

    arq = FakeArq()
    callback = _callback(f"{NOTES_ONLY}:{material_id}")
    async with session_scope(sessions) as session:
        await handle_notes_only(callback, session, arq)  # type: ignore[arg-type]

    assert arq.jobs == []
    assert await _mode_and_status(engine, material_id) == ("full", "processing")
    callback.answer.assert_awaited_once_with(ANSWER_STALE, show_alert=True)  # type: ignore[attr-defined]


async def test_stranger_cannot_act_on_someone_elses_material(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Изоляция по пользователю: чужой `material_id` в кнопке не работает.

    `callback_data` приходит от клиента, и подставить туда чужой
    идентификатор ничто не мешает. Белый список §1.3 от этого не защищает: в
    нём может быть несколько человек, и материалы у них разные.
    """
    material_id = await _parked_material(engine, sessions)
    async with engine.begin() as connection:
        await connection.execute(insert(User).values(tg_id=STRANGER_TG, created_at=NOW))

    arq = FakeArq()
    callback = _callback(f"{NOTES_ONLY}:{material_id}", tg_id=STRANGER_TG)
    async with session_scope(sessions) as session:
        await handle_notes_only(callback, session, arq)  # type: ignore[arg-type]

    assert arq.jobs == []
    assert await _mode_and_status(engine, material_id) == ("full", "processing")


async def test_malformed_callback_data_is_ignored(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Испорченный `callback_data` не роняет обработчик.

    Достижимо: данные приходят от клиента. Падение здесь дало бы пользователю
    «Что-то пошло не так» на нажатие кнопки, которую нарисовал сам бот.
    """
    arq = FakeArq()
    for data in (f"{FULL}", f"{FULL}:abc", "gate:full:1:2"):
        callback = _callback(data)
        async with session_scope(sessions) as session:
            await handle_full(callback, session, arq)  # type: ignore[arg-type]
        callback.answer.assert_awaited_once_with(ANSWER_STALE, show_alert=True)  # type: ignore[attr-defined]
    assert arq.jobs == []
