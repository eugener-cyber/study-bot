"""Выбор вопроса в момент выдачи. ТЗ §13.3, И-2.

Четыре шага отката проверяются по отдельности, и отдельно — два места, где
запрос ломается молча: пустой `direction` и пустой `shown_count`. Оба колонки
`nullable` (§3), а `ORDER BY … DESC` в Postgres по умолчанию `NULLS FIRST`, то
есть строка с пустым значением обходит ту, которая условию удовлетворяет.

Шаги 1 и 2 требуют вопроса **другого типа**, а в WP-05 отвечаемый тип один
(Issue #31). Поэтому в этих тестах множество типов передаётся параметром —
тем самым, которым WP-14 и WP-15 его расширят. Что по умолчанию выдаётся
только `mcq`, проверяет отдельный тест.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.learning.selection import COLD_START_DIFFICULTY_CAP, select_question
from tests.learning_factory import NOW, add_fact, add_question, scaffold

pytestmark = pytest.mark.db

BOTH_TYPES = frozenset({"mcq", "open"})
"""Множество типов, какое будет после WP-15. Нужно, чтобы проверить шаги 1 и 2."""


async def _fact(engine: AsyncEngine) -> int:
    base = await scaffold(engine)
    return await add_fact(engine, base)


def test_cold_start_cap_is_the_first_row_of_the_adaptivity_table() -> None:
    """§15: при `mastery < 0.4` потолок сложности равен 1. Литерал намеренно.

    §14.1 задаёт вырожденный случай прямо: при `seen_facts = 0` mastery равна
    нулю, то есть пользователь без ответов попадает в первую строку §15.
    Значение 3 («без ограничения») противоречило бы таблице.
    """
    assert COLD_START_DIFFICULTY_CAP == 1


async def test_step_one_prefers_other_type_and_other_direction(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Шаг 1: другой тип, другое направление, в пределах потолка."""
    fact_id = await _fact(engine)
    same = await add_question(engine, fact_id, question_type="mcq", direction="forward")
    other = await add_question(engine, fact_id, question_type="open", direction="reverse")

    async with sessions() as session:
        chosen = await select_question(
            session,
            fact_id,
            last_q_type="mcq",
            last_direction="forward",
            answerable_types=BOTH_TYPES,
        )

    assert chosen is not None
    assert chosen.id == other, "выбран вопрос того же типа и направления"
    assert chosen.id != same


async def test_step_two_drops_the_direction_requirement(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Шаг 2: направление совпадает, но тип другой — это лучше, чем ничего."""
    fact_id = await _fact(engine)
    await add_question(engine, fact_id, question_type="mcq", direction="forward")
    other_type = await add_question(engine, fact_id, question_type="open", direction="forward")

    async with sessions() as session:
        chosen = await select_question(
            session,
            fact_id,
            last_q_type="mcq",
            last_direction="forward",
            answerable_types=BOTH_TYPES,
        )

    assert chosen is not None and chosen.id == other_type


async def test_fact_with_two_questions_of_one_type_still_gets_one(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Шаг 3 и приёмка §34: откат обязателен.

    «У факта с двумя вопросами одного типа условие `type ≠ last_q_type` не
    выполнится никогда, и без отката факт выпал бы из повторений навсегда».
    Порядок внутри шага решает `shown_count`.
    """
    fact_id = await _fact(engine)
    shown_often = await add_question(
        engine, fact_id, question_type="mcq", direction="forward", shown_count=5
    )
    shown_rarely = await add_question(
        engine, fact_id, question_type="mcq", direction="reverse", shown_count=1
    )

    async with sessions() as session:
        chosen = await select_question(
            session, fact_id, last_q_type="mcq", last_direction="forward"
        )

    assert chosen is not None
    assert chosen.id == shown_rarely
    assert chosen.id != shown_often


async def test_difficulty_cap_is_relaxed_last(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§13.3: при конфликте приоритет у потолка, а не у разнообразия типов.

    Это правка v3.4→v3.5: прежний порядок снимал ограничение по сложности
    раньше ограничения по типу и противоречил приёмке §34 («вопрос со
    сложностью выше cap не выбирается, если подходящий существует»).
    """
    fact_id = await _fact(engine)
    other_type_too_hard = await add_question(
        engine, fact_id, question_type="open", direction="forward", difficulty=3
    )
    same_type_within_cap = await add_question(
        engine, fact_id, question_type="mcq", direction="reverse", difficulty=1
    )

    async with sessions() as session:
        chosen = await select_question(
            session,
            fact_id,
            last_q_type="mcq",
            last_direction="forward",
            difficulty_cap=1,
            answerable_types=BOTH_TYPES,
        )

    assert chosen is not None
    assert chosen.id == same_type_within_cap
    assert chosen.id != other_type_too_hard


async def test_question_above_cap_is_returned_when_nothing_else_fits(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Шаг 4: потолок снимается, когда под ним пусто.

    Поэтому `cap = 1` не лишает факт заданий — вторая половина теста о
    потолке.
    """
    fact_id = await _fact(engine)
    hard = await add_question(engine, fact_id, difficulty=3)

    async with sessions() as session:
        chosen = await select_question(
            session, fact_id, last_q_type=None, last_direction=None, difficulty_cap=1
        )

    assert chosen is not None and chosen.id == hard


async def test_first_presentation_still_honours_the_cap(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Первое предъявление: `last_q_type` и `last_direction` пусты.

    В плане я обещал, что этот тест красен на наивной формулировке
    `type <> :last_type`. Проверил прогоном — **он зелёный**, и обещание было
    неверным: SQLAlchemy превращает сравнение с `None` в `IS NOT NULL`, то
    есть на пустых привязках наивная запись ведёт себя правильно сама. Дефект
    `NULL`-ключей возникает не на пустой привязке, а на пустой **колонке** —
    это `test_empty_direction_does_not_jump_the_queue`, и он действительно
    красен.

    Тест остаётся: он закрепляет, что потолок сложности действует и на первом
    предъявлении, и страхует от перехода на текстовый SQL, где подстановка
    `NULL` в `<>` правильной сама по себе не будет.
    """
    fact_id = await _fact(engine)
    too_hard_never_shown = await add_question(
        engine, fact_id, question_type="mcq", direction="forward", difficulty=3, shown_count=0
    )
    within_cap_shown_often = await add_question(
        engine, fact_id, question_type="open", direction="forward", difficulty=1, shown_count=5
    )

    async with sessions() as session:
        chosen = await select_question(
            session,
            fact_id,
            last_q_type=None,
            last_direction=None,
            difficulty_cap=1,
            answerable_types=BOTH_TYPES,
        )

    assert chosen is not None
    assert chosen.id == within_cap_shown_often
    assert chosen.id != too_hard_never_shown


async def test_empty_direction_does_not_jump_the_queue(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Пустой `direction` не обходит вопрос, удовлетворяющий шагу 1.

    Красный на формулировке `direction <> :last_dir`: на строке с пустым
    направлением ключ становится `NULL` и при `DESC` встаёт впереди `TRUE`.
    Числа `shown_count` разведены явно (9 и 0), чтобы верная реализация
    решала по ним, а не по совпадению.
    """
    fact_id = await _fact(engine)
    no_direction = await add_question(
        engine, fact_id, question_type="open", direction=None, shown_count=9
    )
    proper = await add_question(
        engine, fact_id, question_type="open", direction="reverse", shown_count=0
    )

    async with sessions() as session:
        chosen = await select_question(
            session,
            fact_id,
            last_q_type="mcq",
            last_direction="forward",
            answerable_types=BOTH_TYPES,
        )

    assert chosen is not None
    assert chosen.id == proper, "вопрос с пустым направлением обошёл очередь"
    assert chosen.id != no_direction


async def test_empty_shown_count_means_never_shown(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Пустой `shown_count` — «не показывали», а не «показывали больше всех».

    Красный на формулировке `shown_count ASC` без `COALESCE`: `ASC` в
    Postgres даёт `NULLS LAST`.
    """
    fact_id = await _fact(engine)
    never = await add_question(
        engine, fact_id, question_type="mcq", direction="forward", shown_count=None
    )
    often = await add_question(
        engine, fact_id, question_type="mcq", direction="reverse", shown_count=5
    )

    async with sessions() as session:
        chosen = await select_question(
            session, fact_id, last_q_type="mcq", last_direction="forward"
        )

    assert chosen is not None
    assert chosen.id == never
    assert chosen.id != often


async def test_ties_are_broken_by_last_shown_at_nulls_first(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§13.3 дословно: `last_shown_at ASC NULLS FIRST`."""
    fact_id = await _fact(engine)
    shown_once = await add_question(
        engine,
        fact_id,
        question_type="mcq",
        direction="forward",
        shown_count=1,
        last_shown_at=NOW - datetime.timedelta(days=1),
    )
    never_shown = await add_question(
        engine,
        fact_id,
        question_type="mcq",
        direction="reverse",
        shown_count=1,
        last_shown_at=None,
    )

    async with sessions() as session:
        chosen = await select_question(
            session, fact_id, last_q_type="mcq", last_direction="forward"
        )

    assert chosen is not None
    assert chosen.id == never_shown
    assert chosen.id != shown_once


@pytest.mark.parametrize("status", ["draft", "rejected"])
async def test_only_valid_questions_are_selected(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    status: str,
) -> None:
    """И-6 и §13.3: отбирается только `status = 'valid'`."""
    fact_id = await _fact(engine)
    await add_question(engine, fact_id, status=status)

    async with sessions() as session:
        chosen = await select_question(session, fact_id, last_q_type=None, last_direction=None)

    assert chosen is None


async def test_open_question_is_not_selected_by_default(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """По умолчанию отбираются только отвечаемые типы — в WP-05 это `mcq`.

    Литерал `"open"` намеренно: тест обязан упасть, когда `open` попадёт в
    реестр обработчиков ответа без своего пакета (Issue #31).
    """
    fact_id = await _fact(engine)
    await add_question(engine, fact_id, question_type="open", direction="forward")

    async with sessions() as session:
        chosen = await select_question(session, fact_id, last_q_type=None, last_direction=None)

    assert chosen is None
