"""Задача воркера: путь отказа и путь ожидания. ТЗ §4.1, §4.5, §37.

Путь отказа не проверял ни один тест: ревью PR #14 вставило
`raise AssertionError` в `except Exception` и прогон не заметил. А это весь
штатный путь §4.1 `processing → failed`: перевод статуса, текст пользователю и
запись **в своей** транзакции, чтобы не быть откаченной вместе со сбоем.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.stub import StubAdapter
from core.db.models import Material, User
from core.db.session import session_scope
from core.ingest.awaiting import is_awaiting, park, progress_message_id, remember_message, resume
from core.storage import MaterialStorage
from tests.test_dispatcher import _settings

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _material(engine: AsyncEngine, tg_id: int = 1200) -> int:
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
                        origin={"path": "/tmp/none.md"},
                        progress={},
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


def _ctx(sessions: async_sessionmaker[AsyncSession], tmp_path: Path) -> dict[str, Any]:
    return {
        "settings": _settings(),
        "sessions": sessions,
        "storage": MaterialStorage(tmp_path),
        "bot": None,
    }


async def _status_and_error(engine: AsyncEngine, material_id: int) -> tuple[str, str | None]:
    async with engine.connect() as connection:
        status, error = (
            await connection.execute(
                select(Material.status, Material.error).where(Material.id == material_id)
            )
        ).one()
    return str(status), error


# --- Путь отказа ----------------------------------------------------------


async def test_failed_task_marks_material_failed_with_a_reason(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой обработки переводит материал в `failed` с текстом (§4.1).

    Три свойства в одном тесте, и все три нужны: статус, текст, и то, что
    исключение поднимается дальше — иначе ARQ сочтёт задачу успешной и не
    запишет отказ в свой журнал.
    """
    import worker

    material_id = await _material(engine)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("стадия сломалась")

    monkeypatch.setattr(worker, "probe_and_gate", boom)

    with pytest.raises(RuntimeError, match="стадия сломалась"):
        await worker.process_material_task(_ctx(sessions, tmp_path), material_id, "/tmp/none.md")

    status, error = await _status_and_error(engine, material_id)
    assert status == "failed"
    assert error is not None
    assert "Обработать заново" in error, "причина не объясняет, что делать"


async def test_failure_record_survives_the_rollback(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mark_failed` пишется в своей транзакции, а не в сбойной.

    Это главное свойство пути отказа. Запись в той же границе была бы
    откачена вместе со сбоем, и материал остался бы в `processing` навсегда —
    состояния «обрабатывается, но никто не обрабатывает» §4.1 не
    предусматривает, и пользователь ждал бы вечно.

    Проверяется тем, что после исключения статус **изменился**: при записи в
    сбойной границе он остался бы прежним.
    """
    import worker

    material_id = await _material(engine)
    assert (await _status_and_error(engine, material_id))[0] == "processing"

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("сбой внутри транзакции стадии")

    monkeypatch.setattr(worker, "process_material", boom)

    with pytest.raises(RuntimeError):
        await worker.process_material_task(_ctx(sessions, tmp_path), material_id, "/tmp/none.md")

    assert (await _status_and_error(engine, material_id))[0] == "failed"


async def test_successful_task_reaches_ready(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обратное направление: при успехе материал доходит до `ready`.

    Без этого теста проверки пути отказа прошли бы и у задачи, которая падает
    всегда.
    """
    import worker

    material_id = await _material(engine)
    monkeypatch.setattr(
        worker, "fragments_from_text_or_stub", lambda path: StubAdapter()._fragments
    )

    result = await worker.process_material_task(
        _ctx(sessions, tmp_path), material_id, "/tmp/none.md"
    )

    assert result == "ready"
    assert (await _status_and_error(engine, material_id))[0] == "ready"


# --- message_id переживает гейт (§4.5) ------------------------------------


async def test_message_id_survives_park_and_resume(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.5 «одно сообщение»: идентификатор не теряется на гейте.

    `park` и `resume` затирают `progress`, и без исключения для `message_id`
    первый же проход через гейт бюджета терял бы сообщение — прогресс после
    подтверждения уходил бы в новое. То есть «одно сообщение» нарушалось бы
    именно в сценарии, для которого ожидание и существует. Нашло ревью PR #14.
    """
    material_id = await _material(engine)

    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 4242)
        assert await progress_message_id(session, material_id) == 4242

    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")
        assert await is_awaiting(session, material_id) is True
        assert (
            await progress_message_id(session, material_id) == 4242
        ), "park потерял идентификатор сообщения"

    async with session_scope(sessions) as session:
        await resume(session, material_id)
        assert await is_awaiting(session, material_id) is False
        assert (
            await progress_message_id(session, material_id) == 4242
        ), "resume потерял идентификатор сообщения"


async def test_park_still_clears_the_previous_question(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Сохранение `message_id` не вернуло остатки прошлого ожидания.

    `resume` затирает `question` и `payload` — иначе пользователь увидел бы
    вопрос о бюджете при выборе языка. Проверяется, что исключение сделано
    ровно для `message_id`, а не для всего `progress`.
    """
    material_id = await _material(engine)

    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 77)
        await park(session, material_id, "Про бюджет", kind="budget", payload={"a": 1})
    async with session_scope(sessions) as session:
        await resume(session, material_id)

    async with engine.connect() as connection:
        progress = (
            await connection.execute(select(Material.progress).where(Material.id == material_id))
        ).scalar_one()

    assert progress == {"message_id": 77}, f"в progress осталось лишнее: {progress}"


async def test_progress_message_id_is_none_for_seeded_material(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """`None` — нормальный исход, а не ошибка.

    Материал мог прийти через `make seed`, где Telegram не участвует вовсе.
    Падение здесь сделало бы харнесс §1.4 неработоспособным.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        assert await progress_message_id(session, material_id) is None
