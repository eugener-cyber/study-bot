"""Материалы: дубликаты, перенос, статусы. ТЗ §4.1, §4.4."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.db.models import Material, Subject, User
from core.db.session import session_scope
from core.services.materials import (
    create_or_find,
    find_duplicate,
    mark_failed,
    mark_ready,
    move_to_subject,
    sha256_of,
)

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _user(engine: AsyncEngine, tg_id: int) -> int:
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
                )
            ).scalar_one()
        )


async def _material_count(engine: AsyncEngine) -> int:
    async with engine.connect() as connection:
        return int(
            (await connection.execute(select(func.count()).select_from(Material))).scalar_one()
        )


def test_sha256_reads_in_chunks(tmp_path: Path) -> None:
    """Хеш считается блоками, а не чтением файла целиком.

    §2.2 поднимает лимит файла до 2 ГБ. Чтение лекционного видео целиком в
    память положило бы процесс — это условие работоспособности на заявленном
    лимите, а не оптимизация. Проверяется на файле больше одного блока.
    """
    import hashlib

    path = tmp_path / "big.bin"
    payload = b"x" * (1024 * 1024 + 7)
    path.write_bytes(payload)

    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


async def test_duplicate_returns_existing_without_copy(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.4: дубль показывает существующий материал, копия не создаётся."""
    user_id = await _user(engine, 800)
    origin = {"sha256": "a" * 64, "orig_name": "lecture.pdf"}

    async with session_scope(sessions) as session:
        first, duplicate = await create_or_find(
            session, user_id=user_id, kind="pdf", title="Лекция", origin=origin
        )
        first_id = first.id
        assert duplicate is False

    async with session_scope(sessions) as session:
        second, duplicate = await create_or_find(
            session, user_id=user_id, kind="pdf", title="Лекция снова", origin=origin
        )
        assert duplicate is True
        assert second.id == first_id

    assert await _material_count(engine) == 1


async def test_duplicate_flag_is_returned_not_inferred(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Признак дубликата возвращается явно.

    §4.4 требует показать существующий материал с кнопками, и хендлер должен
    знать, какой из двух экранов рисовать. Выводить это по косвенным
    признакам — например, по совпадению `created_at` — значило бы угадывать.
    """
    user_id = await _user(engine, 801)
    async with session_scope(sessions) as session:
        _, first = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "b" * 64}
        )
    async with session_scope(sessions) as session:
        _, second = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "b" * 64}
        )
    assert (first, second) == (False, True)


async def test_same_file_for_two_users_is_two_materials(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Дубликат ищется в рамках пользователя, а не глобально (§4.4).

    Один и тот же файл у двух людей — два материала с отдельной историей
    обучения. Глобальный поиск дубликата отдал бы второму пользователю
    материал первого вместе с его фактами и расписанием.
    """
    first_user = await _user(engine, 802)
    second_user = await _user(engine, 803)
    origin = {"sha256": "c" * 64}

    async with session_scope(sessions) as session:
        await create_or_find(session, user_id=first_user, kind="pdf", title="X", origin=origin)
    async with session_scope(sessions) as session:
        _, duplicate = await create_or_find(
            session, user_id=second_user, kind="pdf", title="X", origin=origin
        )
        assert duplicate is False

    assert await _material_count(engine) == 2


async def test_material_without_sha256_is_never_a_duplicate(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Ссылка и текст дубликатом не считаются.

    Уникальный индекс §3.1 частичный по наличию ключа `sha256`. Проверять
    ссылки на совпадение по URL спецификация не требует, и выдумывать это
    правило здесь нельзя (§40).
    """
    user_id = await _user(engine, 804)
    origin = {"url": "https://example.org/page"}

    async with session_scope(sessions) as session:
        await create_or_find(session, user_id=user_id, kind="web", title="A", origin=origin)
    async with session_scope(sessions) as session:
        _, duplicate = await create_or_find(
            session, user_id=user_id, kind="web", title="A", origin=origin
        )
        assert duplicate is False

    assert await _material_count(engine) == 2


async def test_find_duplicate_returns_none_for_unknown_hash(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    user_id = await _user(engine, 805)
    async with session_scope(sessions) as session:
        assert await find_duplicate(session, user_id, "d" * 64) is None


async def test_move_changes_only_the_subject(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.4: перенос меняет только `subject_id`.

    Ни статуса, ни `completed_stages`, ни артефактов: перенос — смена папки, а
    не переобработка. Обратное означало бы потерю истории обучения при
    нажатии кнопки, которая выглядит безобидной. Поэтому ассерт не только на
    новый предмет, но и на неизменность остального.
    """
    user_id = await _user(engine, 806)
    async with engine.begin() as connection:
        subject_id = (
            await connection.execute(
                insert(Subject)
                .values(user_id=user_id, title="Химия", created_at=NOW)
                .returning(Subject.id)
            )
        ).scalar_one()

    async with session_scope(sessions) as session:
        material, _ = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "e" * 64}
        )
        material_id = material.id
        material.status = "ready"
        material.completed_stages = ["extract", "sections"]

    async with session_scope(sessions) as session:
        await move_to_subject(session, material_id, int(subject_id))

    async with engine.connect() as connection:
        row = (
            await connection.execute(
                select(Material.subject_id, Material.status, Material.completed_stages).where(
                    Material.id == material_id
                )
            )
        ).one()
    assert row[0] == subject_id
    assert row[1] == "ready", "перенос изменил статус"
    assert list(row[2]) == ["extract", "sections"], "перенос сбросил стадии"


async def test_mark_ready_sets_processed_at_and_clears_progress(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    user_id = await _user(engine, 807)
    async with session_scope(sessions) as session:
        material, _ = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "f" * 64}
        )
        material_id = material.id

    async with session_scope(sessions) as session:
        await mark_ready(session, material_id)

    async with engine.connect() as connection:
        status, processed_at, progress = (
            await connection.execute(
                select(Material.status, Material.processed_at, Material.progress).where(
                    Material.id == material_id
                )
            )
        ).one()
    assert status == "ready"
    assert processed_at is not None
    assert not progress


async def test_mark_failed_requires_a_reason(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Причина — обязательный параметр, а не необязательный.

    Материал в `failed` без текста выглядит сбоем системы, и пользователь
    будет искать поломку вместо того, чтобы нажать «Обработать заново».
    """
    user_id = await _user(engine, 808)
    async with session_scope(sessions) as session:
        material, _ = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "0" * 64}
        )
        material_id = material.id

    async with session_scope(sessions) as session:
        await mark_failed(session, material_id, "Не удалось прочитать файл")

    async with engine.connect() as connection:
        status, error = (
            await connection.execute(
                select(Material.status, Material.error).where(Material.id == material_id)
            )
        ).one()
    assert status == "failed"
    assert error == "Не удалось прочитать файл"


async def test_new_material_starts_queued_with_empty_stages(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.1: жизненный цикл начинается с `queued`.

    И с пустым `completed_stages`: непустой список при создании означал бы,
    что возобновление §4.3 пропустит стадии, которых не было.
    """
    user_id = await _user(engine, 809)
    async with session_scope(sessions) as session:
        material, _ = await create_or_find(
            session, user_id=user_id, kind="pdf", title="X", origin={"sha256": "1" * 64}
        )
        assert material.status == "queued"
        assert list(material.completed_stages or []) == []
        assert material.mode == "full"
