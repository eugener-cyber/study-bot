"""Транзакционная граница. ТЗ §4.2, чек-лист ревью раздел 2.

Проверяется то, что `core/db/session.py` обещает докстрокой: «частичный
результат при сбое посередине невозможен». Обещание до этого файла не держал
ни один тест — подмена `rollback()` на `commit()` оставляла прогон зелёным
(242 passed). Нашло ревью PR #10.

Тесты работают с **настоящей** сессией и настоящей базой. `BrokenSession` из
`test_consent.py` для этого не годится: у него `rollback()` — пустая
заглушка, то есть сама граница в проверке не участвует.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.db.models import User
from core.db.session import session_scope

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _user_count(engine: AsyncEngine) -> int:
    async with engine.connect() as connection:
        return int((await connection.execute(select(func.count()).select_from(User))).scalar_one())


async def test_successful_scope_commits(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Обратное направление: при успехе данные сохраняются.

    Без этого теста проверка откатa прошла бы и у границы, которая откатывает
    всегда, — то есть у неработающей записи.
    """
    async with session_scope(sessions) as session:
        session.add(User(tg_id=2001, created_at=NOW))

    assert await _user_count(engine) == 1


async def test_failure_leaves_no_partial_result(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Главный тест файла: сбой посередине не оставляет данных.

    Первая запись проходит и даже сбрасывается в базу через `flush`, вторая
    операция падает. Если граница коммитит вместо откатa, первая строка
    останется — и материал будет наполовину обработан при стадии, которая
    формально не выполнена (§4.2).
    """
    with pytest.raises(RuntimeError, match="сбой посередине"):
        async with session_scope(sessions) as session:
            session.add(User(tg_id=2002, created_at=NOW))
            await session.flush()
            raise RuntimeError("сбой посередине")

    assert await _user_count(engine) == 0, "частичный результат сохранён"


async def test_exception_is_not_swallowed(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Исключение поднимается дальше, а не гасится.

    Проглоченная здесь ошибка означала бы, что пользователь получил ответ об
    успехе при незаписанных данных. Конвейер на этом держится: `run_stage`
    ловит исключение снаружи границы и не ставит отметку о выполнении стадии.
    """
    with pytest.raises(ValueError, match="наружу"):
        async with session_scope(sessions) as session:
            session.add(User(tg_id=2003, created_at=NOW))
            raise ValueError("наружу")


async def test_rollback_covers_writes_made_before_the_failure(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Откат снимает все записи границы, а не только последнюю.

    Три вставки, сбой на четвёртом действии. Граница, откатывающая частично,
    оставила бы материал в состоянии, которого нет ни в одном статусе §4.1.
    """
    with pytest.raises(RuntimeError):
        async with session_scope(sessions) as session:
            for tg_id in (2004, 2005, 2006):
                session.add(User(tg_id=tg_id, created_at=NOW))
                await session.flush()
            raise RuntimeError("после трёх вставок")

    assert await _user_count(engine) == 0


async def test_constraint_violation_also_rolls_back(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Нарушение ограничения базы — тот же путь, что исключение в коде.

    Отдельный тест, потому что путь другой: `IntegrityError` поднимает сам
    драйвер, и транзакция к этому моменту уже помечена как сбойная. Схема
    §3.1 опирается на такие нарушения — второй активный сеанс, дубль
    материала, занятый слот вопроса, — и граница обязана их выдерживать.
    """
    from sqlalchemy.exc import IntegrityError

    async with session_scope(sessions) as session:
        session.add(User(tg_id=2007, created_at=NOW))

    with pytest.raises(IntegrityError):
        async with session_scope(sessions) as session:
            session.add(User(tg_id=2008, created_at=NOW))
            await session.flush()
            session.add(User(tg_id=2007, created_at=NOW))
            await session.flush()

    assert await _user_count(engine) == 1, "вставка до нарушения не откачена"
