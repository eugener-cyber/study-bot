"""Минимальный планировщик WP-05. ТЗ §16, §18, §30.3.

Проверяется и то, что планировщик делает, и то, чего он сознательно **не**
делает: корзины, interleaving и дневные нормы §16 относятся к WP-13, и их
отсутствие здесь не дефект. Зато порог `MIN_SESSION_ITEMS`, предикат
обучаемости внутри запроса и отбор только отвечаемых типов — его работа.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.config import Settings
from core.db.models import SessionItem, User
from core.services.sessions import assemble
from tests.conftest import ENV_STUB
from tests.learning_factory import NOW, Scaffold, add_fact, add_question, add_review_state, scaffold

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parent.parent

MIN_SESSION_ITEMS = 3
"""§30.3: `MIN_SESSION_ITEMS` = 3. Литерал намеренно — урок про `TRUNCATE_AT`."""


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, **ENV_STUB)  # type: ignore[arg-type]


def test_spec_minimum_matches_settings(settings: Settings) -> None:
    """Литерал теста совпадает с умолчанием конфигурации."""
    assert settings.MIN_SESSION_ITEMS == MIN_SESSION_ITEMS


async def _due_facts(
    engine: AsyncEngine,
    base: Scaffold,
    count: int,
    *,
    question_type: str = "mcq",
) -> list[int]:
    """`count` обучаемых фактов, которым пора, с валидным вопросом у каждого."""
    created: list[int] = []
    for index in range(count):
        fact_id = await add_fact(engine, base, statement=f"Факт {index}")
        await add_question(engine, fact_id, question_type=question_type)
        await add_review_state(engine, base.user_id, fact_id)
        created.append(fact_id)
    return created


async def test_session_is_assembled_from_due_facts(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    base = await scaffold(engine)
    facts = await _due_facts(engine, base, MIN_SESSION_ITEMS)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()

    assert session_id is not None

    async with sessions() as session:
        rows = (
            await session.execute(
                select(SessionItem.ord, SessionItem.fact_id)
                .where(SessionItem.session_id == session_id)
                .order_by(SessionItem.ord)
            )
        ).all()

    assert [int(row[0]) for row in rows] == [1, 2, 3]
    assert {int(row[1]) for row in rows} == set(facts)


async def test_too_few_candidates_give_no_session(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§16: «Если кандидатов меньше `MIN_SESSION_ITEMS`, сессия не создаётся»."""
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS - 1)

    async with sessions() as session:
        assert await assemble(session, user_id=base.user_id, settings=settings, now=NOW) is None


async def test_suspended_state_is_not_a_candidate(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§16: `NOT rs.suspended` в общем фильтре пригодности."""
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS)
    extra = await add_fact(engine, base, statement="Приостановленный")
    await add_question(engine, extra)
    await add_review_state(engine, base.user_id, extra, suspended=True)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        # Положительный ассерт до отрицательного: сессия действительно
        # собралась. Без него «лишнего факта в сессии нет» выполняется и у
        # планировщика, который не выдаёт ничего вовсе.
        assert session_id is not None, "сессия не собралась — проверять нечего"
        facts = (
            (
                await session.execute(
                    select(SessionItem.fact_id).where(SessionItem.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )

    selected = {int(item) for item in facts}
    assert len(selected) == MIN_SESSION_ITEMS, "сессия собрана не из ожидаемых заданий"
    assert extra not in selected


async def test_derived_fact_is_not_a_candidate(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """И-10 и §14.3: вывод модели не становится источником обучения.

    Факт исключается **предикатом обучаемости**, а не условием планировщика:
    своего условия у планировщика нет вовсе, и это проверяет
    `test_threshold_not_inlined_by_consumers`.
    """
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS)
    derived = await add_fact(engine, base, statement="Вывод модели", derived=True)
    await add_question(engine, derived)
    await add_review_state(engine, base.user_id, derived)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        # Положительный ассерт до отрицательного: сессия действительно
        # собралась. Без него «лишнего факта в сессии нет» выполняется и у
        # планировщика, который не выдаёт ничего вовсе.
        assert session_id is not None, "сессия не собралась — проверять нечего"
        facts = (
            (
                await session.execute(
                    select(SessionItem.fact_id).where(SessionItem.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )

    selected = {int(item) for item in facts}
    assert len(selected) == MIN_SESSION_ITEMS, "сессия собрана не из ожидаемых заданий"
    assert derived not in selected


async def test_fact_with_only_an_open_question_is_skipped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Issue #31: факт обучаем, но задания в WP-05 не даёт.

    Он проходит предикат §14.3 — валидный вопрос у него есть, — и именно
    поэтому фильтр по отвечаемому типу обязан стоять в запросе кандидатов:
    иначе он попал бы в счёт `MIN_SESSION_ITEMS`, а сессия вышла бы короче
    порога.
    """
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS)
    open_only = await add_fact(engine, base, statement="Только открытый вопрос")
    await add_question(engine, open_only, question_type="open")
    await add_review_state(engine, base.user_id, open_only)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        # Положительный ассерт до отрицательного: сессия действительно
        # собралась. Без него «лишнего факта в сессии нет» выполняется и у
        # планировщика, который не выдаёт ничего вовсе.
        assert session_id is not None, "сессия не собралась — проверять нечего"
        facts = (
            (
                await session.execute(
                    select(SessionItem.fact_id).where(SessionItem.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )

    selected = {int(item) for item in facts}
    assert len(selected) == MIN_SESSION_ITEMS, "сессия собрана не из ожидаемых заданий"
    assert open_only not in selected


async def test_open_only_facts_do_not_count_towards_the_minimum(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Вторая половина предыдущего: трёх `open`-фактов не хватает на сессию.

    Без фильтра в запросе кандидатов их было бы ровно `MIN_SESSION_ITEMS`,
    сессия создалась бы — и встала на первом задании, потому что ответить на
    `open` до WP-15 нечем.
    """
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS, question_type="open")

    async with sessions() as session:
        assert await assemble(session, user_id=base.user_id, settings=settings, now=NOW) is None


async def test_future_due_facts_are_not_candidates(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§0.2 и §16: досрочные повторения запрещены.

    Корзина 3 (слабые темы с `due_at` в пределах суток) — WP-13; до неё
    будущий срок означает «не сейчас».
    """
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS)
    later = await add_fact(engine, base, statement="Ещё не пора")
    await add_question(engine, later)
    await add_review_state(engine, base.user_id, later, due_at=NOW + datetime.timedelta(hours=2))

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        # Положительный ассерт до отрицательного: сессия действительно
        # собралась. Без него «лишнего факта в сессии нет» выполняется и у
        # планировщика, который не выдаёт ничего вовсе.
        assert session_id is not None, "сессия не собралась — проверять нечего"
        facts = (
            (
                await session.execute(
                    select(SessionItem.fact_id).where(SessionItem.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )

    selected = {int(item) for item in facts}
    assert len(selected) == MIN_SESSION_ITEMS, "сессия собрана не из ожидаемых заданий"
    assert later not in selected


async def test_facts_with_more_lapses_come_first(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§16, корзина 2: `lapses DESC, due_at ASC`.

    «Факты, на которых человек регулярно ошибается, идут первыми».
    """
    base = await scaffold(engine)
    forgotten = await add_fact(engine, base, statement="Забывается")
    await add_question(engine, forgotten)
    await add_review_state(engine, base.user_id, forgotten, lapses=3)

    easy = await add_fact(engine, base, statement="Идёт легко")
    await add_question(engine, easy)
    await add_review_state(engine, base.user_id, easy, lapses=0)

    middle = await add_fact(engine, base, statement="Середина")
    await add_question(engine, middle)
    await add_review_state(engine, base.user_id, middle, lapses=1)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        ordered = (
            (
                await session.execute(
                    select(SessionItem.fact_id)
                    .where(SessionItem.session_id == session_id)
                    .order_by(SessionItem.ord)
                )
            )
            .scalars()
            .all()
        )

    assert [int(item) for item in ordered] == [forgotten, middle, easy]


async def test_session_size_limits_the_number_of_items(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Размер сессии — `users.session_size` (§3), а не константа §30.3."""
    base = await scaffold(engine)
    await _due_facts(engine, base, 6)
    async with engine.begin() as connection:
        await connection.execute(update(User).where(User.id == base.user_id).values(session_size=4))

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
        count = (
            (
                await session.execute(
                    select(SessionItem.id).where(SessionItem.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )

    assert len(count) == 4


async def test_existing_active_session_is_returned_instead_of_a_second_one(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§18 и §38 №2: двух активных сессий одного пользователя не бывает.

    Положительный ассерт стоит первым: сессия действительно создалась, и
    только потом проверяется, что вторая попытка вернула ту же.
    """
    base = await scaffold(engine)
    await _due_facts(engine, base, MIN_SESSION_ITEMS)

    async with sessions() as session:
        first = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
    assert first is not None, "первая сессия не собралась — проверять нечего"

    async with sessions() as session:
        second = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()

    assert second == first


def test_default_session_size_matches_the_schema() -> None:
    """`DEFAULT_SESSION_SIZE` повторяет `server_default` колонки §3.

    Два числа, означающих одно, разъехались бы молча: планировщик взял бы
    своё, база — своё, и размер сессии зависел бы от того, кто создал строку
    пользователя.
    """
    from core.db.models import User as UserModel
    from core.learning.planner import DEFAULT_SESSION_SIZE

    column = UserModel.__table__.c.session_size
    assert column.server_default is not None
    assert str(column.server_default.arg) == str(DEFAULT_SESSION_SIZE)


def test_candidates_query_keeps_the_partial_index_applicable() -> None:
    """Условие приостановки записано формой, которая не теряет индекс §3.1.

    Частичный индекс `ix_review_states_user_due` объявлен с предикатом
    `WHERE NOT suspended`, и применим он только к условиям, из которых
    Postgres умеет вывести этот предикат. Замер на 200 тысячах строк с
    `enable_seqscan = off` (воспроизводится вручную, в тесте не повторяется —
    на малых таблицах планировщик выбирает перебор при любой формулировке, и
    такой тест был бы зелёным на сломанном коде):

    ```
    NOT suspended       -> Index Scan using ix_review_states_user_due,
                           Index Cond: user_id = 1 AND due_at <= now()
    suspended = false   -> то же
    suspended IS FALSE  -> Seq Scan, индекс не применён
    ```

    Поэтому проверка идёт по исходному тексту — тот же приём, что у
    `test_create_all_is_not_used` и `test_threshold_not_inlined_by_consumers`:
    договорённость, которую не проверяет прогон, живёт до первого спешного
    коммита. Замечание 1 ревью PR #34 нашло здесь `is_not(True)`; `is_(False)`,
    которое выглядит исправлением, теряет индекс точно так же.
    """
    source = (ROOT / "core" / "learning" / "planner.py").read_text(encoding="utf-8")
    condition_lines = [
        line.strip()
        for line in source.splitlines()
        if "suspended" in line and not line.strip().startswith("#")
    ]

    assert condition_lines == ["~ReviewState.suspended,"], condition_lines
    assert "suspended.is_(False)" not in source
    assert "suspended.is_not(True)" not in source
