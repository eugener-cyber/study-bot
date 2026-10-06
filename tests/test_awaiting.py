"""Ожидание ответа пользователя посреди конвейера. ТЗ §4.5."""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.db.models import Material, User
from core.db.session import session_scope
from core.ingest.awaiting import (
    AWAITING_STAGE,
    awaiting_kind,
    expire_stale,
    is_awaiting,
    park,
    resume,
)
from tests.test_dispatcher import _settings

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _material(engine: AsyncEngine, tg_id: int = 700) -> int:
    async with engine.begin() as connection:
        user_id = (
            await connection.execute(
                insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
            )
        ).scalar_one()
        return int(
            (
                await connection.execute(
                    insert(Material)
                    .values(
                        user_id=user_id,
                        kind="text",
                        status="processing",
                        mode="full",
                        completed_stages=[],
                        progress={},
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


async def _row(engine: AsyncEngine, material_id: int) -> tuple[str, dict[str, object], str | None]:
    async with engine.connect() as connection:
        status, progress, error = (
            await connection.execute(
                select(Material.status, Material.progress, Material.error).where(
                    Material.id == material_id
                )
            )
        ).one()
    return str(status), dict(progress or {}), error


async def test_park_keeps_status_processing(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.5: материал остаётся в `processing`, а не получает новый статус.

    §4.1 перечисляет четыре статуса. Пятый сломал бы все выборки по `status`,
    написанные по спецификации.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")

    status, progress, _ = await _row(engine, material_id)
    assert status == "processing"
    assert progress["stage"] == AWAITING_STAGE
    assert progress["question"] == "Материал большой…"


async def test_park_records_what_is_awaited(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """`kind` различает бюджет (§6.3) и выбор языка (§5.4).

    Без него обработчик нажатия не знал бы, что делать с ответом, и пришлось
    бы угадывать по тексту вопроса — то есть по строке, предназначенной
    человеку.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await park(session, material_id, "Какой язык?", kind="language")
        assert await awaiting_kind(session, material_id) == "language"


async def test_resume_clears_progress_entirely(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """`progress` затирается целиком, а не правится поле `stage`.

    Остатки `question` и `payload` от прошлого ожидания попали бы в следующее,
    и пользователь увидел бы вопрос о бюджете при выборе языка.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await park(
            session, material_id, "Материал большой…", kind="budget", payload={"est": "9.99"}
        )
    async with session_scope(sessions) as session:
        await resume(session, material_id)
        assert await is_awaiting(session, material_id) is False

    _, progress, _ = await _row(engine, material_id)
    assert progress == {}


async def test_awaiting_kind_is_none_when_not_awaiting(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        assert await awaiting_kind(session, material_id) is None


async def test_stale_awaiting_becomes_failed_with_a_reason(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.5: через `AWAITING_USER_TIMEOUT_H` материал переводится в `failed`
    с понятной причиной.

    «Понятная причина» — не формальность: материал в `failed` без текста
    выглядит сбоем системы, хотя сбоя не было, и пользователь будет искать
    поломку вместо того, чтобы нажать кнопку.
    """
    settings = _settings()
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")

    long_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        hours=settings.AWAITING_USER_TIMEOUT_H + 1
    )
    async with engine.begin() as connection:
        await connection.execute(
            update(Material)
            .where(Material.id == material_id)
            .values(
                progress={
                    "stage": AWAITING_STAGE,
                    "question": "Материал большой…",
                    "kind": "budget",
                    "since": long_ago.isoformat(),
                }
            )
        )

    async with session_scope(sessions) as session:
        expired = await expire_stale(session, settings)

    assert expired == [material_id]
    status, progress, error = await _row(engine, material_id)
    assert status == "failed"
    assert progress == {}
    assert error is not None
    assert "Обработать заново" in error, "причина не объясняет, что делать"
    assert str(settings.AWAITING_USER_TIMEOUT_H) in error


async def test_fresh_awaiting_is_not_expired(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Материал, поставленный на ожидание только что, не трогается.

    Обратное направление к предыдущему тесту: проверка только «истёкший
    истекает» прошла бы и у функции, переводящей в `failed` всех без разбора.
    """
    settings = _settings()
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")
        assert await expire_stale(session, settings) == []

    status, _, _ = await _row(engine, material_id)
    assert status == "processing"


async def test_expire_ignores_materials_not_awaiting(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Материал в обработке без ожидания не истекает.

    Иначе долгая обработка видео переводилась бы в `failed` по тайм-ауту,
    предназначенному для вопроса пользователю.
    """
    settings = _settings()
    material_id = await _material(engine)

    async with session_scope(sessions) as session:
        assert await expire_stale(session, settings) == []

    status, _, _ = await _row(engine, material_id)
    assert status == "processing"


async def test_expire_tolerates_missing_since(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Ожидание без метки времени не истекает и не роняет уборку.

    Достижимо при ручной правке базы или при миграции данных. Падение здесь
    остановило бы истечение всех остальных материалов — один испорченный
    `progress` заблокировал бы механизм целиком.
    """
    settings = _settings()
    material_id = await _material(engine)
    async with engine.begin() as connection:
        await connection.execute(
            update(Material)
            .where(Material.id == material_id)
            .values(progress={"stage": AWAITING_STAGE, "question": "?"})
        )

    async with session_scope(sessions) as session:
        assert await expire_stale(session, settings) == []
