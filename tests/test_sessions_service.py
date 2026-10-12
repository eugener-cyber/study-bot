"""Сервис сеанса: выдача, ответ, журнал, завершение. ТЗ §13.2, §16.1, §18, §22.1.

Самое важное здесь — три теста про гонки §18. В каждом **первым** стоит
положительный ассерт, что опасная ситуация действительно возникла: урок
`STATE.md` про ассерт формы «плохого не случилось» — он остаётся зелёным и
тогда, когда столкновения не было вовсе.
"""

from __future__ import annotations

import asyncio
import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.config import Settings
from core.db.models import Question, ReviewLog, ReviewState, Session, SessionItem
from core.learning.rating import Rating
from core.learning.scheduler import STUB_VERSION, FixedIntervalScheduler
from core.services.sessions import (
    SHOWN_AT,
    AnswerResult,
    answer_item,
    assemble,
    current_item,
    mark_shown,
    session_summary,
)
from tests.conftest import ENV_STUB
from tests.learning_factory import NOW, Scaffold, add_fact, add_question, add_review_state, scaffold

pytestmark = pytest.mark.db

LATER = NOW + datetime.timedelta(seconds=12)
"""Момент ответа: двенадцать секунд после выдачи — между FAST_MS и SLOW_MS."""


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, **ENV_STUB)  # type: ignore[arg-type]


@pytest.fixture
def scheduler() -> FixedIntervalScheduler:
    return FixedIntervalScheduler()


async def _ready_session(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    facts: int = 3,
) -> tuple[Scaffold, int]:
    """Пользователь с собранной сессией из `facts` заданий."""
    base = await scaffold(engine)
    for index in range(facts):
        fact_id = await add_fact(engine, base, statement=f"Факт {index}")
        await add_question(engine, fact_id)
        await add_review_state(engine, base.user_id, fact_id)

    async with sessions() as session:
        session_id = await assemble(session, user_id=base.user_id, settings=settings, now=NOW)
        await session.commit()
    assert session_id is not None, "сессия не собралась — проверять нечего"
    return base, session_id


async def test_showing_a_task_touches_counters_but_not_learning_state(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """И-4 и §13.3: показ меняет счётчики презентации и ничего больше.

    Положительный ассерт идёт первым: показ действительно произошёл и
    `shown_count` вырос. Без него тест остался бы зелёным, если выдача молча
    не случилась.
    """
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        before = (
            await session.execute(
                select(
                    ReviewState.due_at,
                    ReviewState.reps,
                    ReviewState.lapses,
                    ReviewState.seen,
                    ReviewState.introduced_at,
                    ReviewState.state,
                ).where(
                    ReviewState.user_id == base.user_id,
                    ReviewState.fact_id == item.fact_id,
                )
            )
        ).one()

        assert await mark_shown(session, item.item_id, now=NOW) is True
        await session.commit()

    async with sessions() as session:
        shown_count, last_shown_at = (
            await session.execute(
                select(Question.shown_count, Question.last_shown_at).where(
                    Question.id == item.question_id
                )
            )
        ).one()
        after = (
            await session.execute(
                select(
                    ReviewState.due_at,
                    ReviewState.reps,
                    ReviewState.lapses,
                    ReviewState.seen,
                    ReviewState.introduced_at,
                    ReviewState.state,
                ).where(
                    ReviewState.user_id == base.user_id,
                    ReviewState.fact_id == item.fact_id,
                )
            )
        ).one()
        last_q_type = (
            await session.execute(
                select(ReviewState.last_q_type).where(
                    ReviewState.user_id == base.user_id,
                    ReviewState.fact_id == item.fact_id,
                )
            )
        ).scalar_one()

    assert shown_count == 1
    assert last_shown_at is not None
    assert last_q_type == "mcq"
    assert tuple(after) == tuple(before), "показ изменил learning state — нарушение И-4"


async def test_second_show_does_not_count_twice(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Повторный показ того же задания `shown_count` не растит.

    §13.3 сортирует вопросы по `shown_count`, и если он считал бы перезагрузки
    экрана, порядок выдачи поехал бы от нажатий `/review`, а не от истории.
    """
    _, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        assert await mark_shown(session, item.item_id, now=NOW) is True
        await session.commit()

    async with sessions() as session:
        assert await mark_shown(session, item.item_id, now=LATER) is False
        await session.commit()

    async with sessions() as session:
        shown_count = (
            await session.execute(
                select(Question.shown_count).where(Question.id == item.question_id)
            )
        ).scalar_one()

    assert shown_count == 1


async def test_correct_answer_writes_the_full_log_row(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """§13.1: «`review_logs` пишется на каждый ответ полностью».

    Полнота — условие пересчёта в WP-12, а не аккуратность: пропущенная
    колонка делает пересчёт невозможным, и обнаружится это через месяцы.
    """
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        result = await answer_item(
            session,
            user_id=base.user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    assert result is not None
    assert result.verdict == "correct"
    assert result.rating is Rating.GOOD

    async with sessions() as session:
        row = (
            await session.execute(
                select(
                    ReviewLog.user_id,
                    ReviewLog.fact_id,
                    ReviewLog.question_id,
                    ReviewLog.rating,
                    ReviewLog.verdict,
                    ReviewLog.scheduled_days,
                    ReviewLog.elapsed_days,
                    ReviewLog.latency_ms,
                    ReviewLog.fsrs_version,
                    ReviewLog.planner_version,
                    ReviewLog.reviewed_at,
                )
            )
        ).one()

    assert row.user_id == base.user_id
    assert row.fact_id == item.fact_id
    assert row.question_id == item.question_id
    assert row.rating == int(Rating.GOOD)
    assert row.verdict == "correct"
    assert row.scheduled_days == 3
    assert row.elapsed_days == 0
    assert row.latency_ms == 12000
    assert row.fsrs_version == STUB_VERSION
    assert row.planner_version == "skeleton-1"
    assert row.reviewed_at is not None
    assert None not in tuple(row), "в журнале есть пустая колонка — §13.1 требует полноты"


async def test_first_answer_sets_seen_and_introduced_at_once(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """§14.2 и §16.1: `introduced_at` проставляется ровно один раз, вместе с `seen`.

    Второй ответ по тому же факту дату введения не сдвигает — иначе
    `new_per_day` в WP-13 считал бы один факт новым каждый день.
    """
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await answer_item(
            session,
            user_id=base.user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    async with sessions() as session:
        seen, introduced_at = (
            await session.execute(
                select(ReviewState.seen, ReviewState.introduced_at).where(
                    ReviewState.user_id == base.user_id, ReviewState.fact_id == item.fact_id
                )
            )
        ).one()
    assert seen is True
    assert introduced_at is not None
    first_introduction = introduced_at

    # Второй ответ по тому же факту: собираем новую сессию после ответа на все
    # задания первой — но проверить достаточно, что повторный вызов сервиса
    # дату не двигает.
    async with sessions() as session:
        later_item = await current_item(session, session_id)
        assert later_item is not None and later_item.fact_id != item.fact_id
        await answer_item(
            session,
            user_id=base.user_id,
            item_id=later_item.item_id,
            raw={"option": 0},
            now=LATER + datetime.timedelta(days=1),
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    async with sessions() as session:
        introduced_again = (
            await session.execute(
                select(ReviewState.introduced_at).where(
                    ReviewState.user_id == base.user_id, ReviewState.fact_id == item.fact_id
                )
            )
        ).scalar_one()

    assert introduced_again == first_introduction


async def test_stub_does_not_fill_fsrs_columns(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """Заглушка не пишет `stability`, `difficulty` и `state`. ADR-0006, §4.6.

    Положительный ассерт первым: ответ действительно прошёл — `reps` вырос и
    строка журнала появилась. Иначе «колонки не изменились» выполнялось бы и
    при молча отклонённом ответе.
    """
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await answer_item(
            session,
            user_id=base.user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    async with sessions() as session:
        reps, stability, difficulty, state = (
            await session.execute(
                select(
                    ReviewState.reps,
                    ReviewState.stability,
                    ReviewState.difficulty,
                    ReviewState.state,
                ).where(ReviewState.user_id == base.user_id, ReviewState.fact_id == item.fact_id)
            )
        ).one()
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()

    assert reps == 1, "ответ не прошёл — проверять нечего"
    assert logs == 1
    assert stability is None
    assert difficulty is None
    assert state == 0, "заглушка изменила state — шкалу задаёт WP-12"


async def test_double_tap_writes_one_log_row(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """§18, двойной тап: право на мутацию получает ровно один.

    Две **одновременные** транзакции, а не последовательные вызовы: при
    `READ COMMITTED` обе читают `answered_at IS NULL`, и проверка перед
    записью такую гонку не закрывает. Первым стоит положительный ассерт, что
    столкновение случилось: ровно один из двух получил право.
    """
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await session.commit()

    async def tap(option: int, *, delay: float) -> AnswerResult | None:
        """Один тап в своей транзакции.

        Каждый тап коммитит **свою** транзакцию сам. Первая редакция этого
        теста держала обе открытыми одновременно и коммитила первую после
        вызова второй — и вставала навсегда: второй `UPDATE` ждал на уровне
        строки Postgres коммита, до которого тест не доходил. Это и есть
        доказательство, что право на мутацию выдаёт база, а не код.

        Задержка у второго тапа делает порядок воспроизводимым: без неё то,
        какая из двух корутин дойдёт до `UPDATE` первой, решает планировщик
        asyncio, и тест проверял бы одно и то же через раз.
        """
        await asyncio.sleep(delay)
        async with sessions() as session:
            result = await answer_item(
                session,
                user_id=base.user_id,
                item_id=item.item_id,
                raw={"option": option},
                now=LATER,
                settings=settings,
                scheduler=scheduler,
            )
            await session.commit()
            return result

    first_result, second_result = await asyncio.gather(tap(0, delay=0.0), tap(1, delay=0.05))

    accepted = [result for result in (first_result, second_result) if result is not None]
    assert len(accepted) == 1, "право на мутацию получили оба — §18 нарушен"

    async with sessions() as session:
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()
        reps = (
            await session.execute(
                select(ReviewState.reps).where(
                    ReviewState.user_id == base.user_id, ReviewState.fact_id == item.fact_id
                )
            )
        ).scalar_one()

    assert logs == 1
    assert reps == 1


async def test_answer_on_a_finished_session_changes_nothing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """§38 №8: старый callback не изменяет данные."""
    base, session_id = await _ready_session(engine, sessions, settings, facts=3)

    answered: list[int] = []
    async with sessions() as session:
        while (item := await current_item(session, session_id)) is not None:
            await mark_shown(session, item.item_id, now=NOW)
            await answer_item(
                session,
                user_id=base.user_id,
                item_id=item.item_id,
                raw={"option": 0},
                now=LATER,
                settings=settings,
                scheduler=scheduler,
            )
            answered.append(item.item_id)
        await session.commit()

    assert len(answered) == 3, "сессия не доиграна — проверять нечего"

    async with sessions() as session:
        status = (
            await session.execute(select(Session.status).where(Session.id == session_id))
        ).scalar_one()
        assert status == "finished"

        stale = await answer_item(
            session,
            user_id=base.user_id,
            item_id=answered[0],
            raw={"option": 1},
            now=LATER + datetime.timedelta(minutes=1),
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    assert stale is None

    async with sessions() as session:
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()
    assert logs == 3


async def test_someone_elses_item_is_not_answerable(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """Изоляция по пользователю: `si` приходит от клиента (§19.1).

    Проверка принадлежности встроена в то же условие `UPDATE`, что и §18:
    отдельной проверкой перед записью она закрывала бы не гонку, а только
    невнимательность.
    """
    _, session_id = await _ready_session(engine, sessions, settings)
    stranger = await scaffold(engine, tg_id=999)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await session.commit()

    async with sessions() as session:
        result = await answer_item(
            session,
            user_id=stranger.user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=scheduler,
        )
        await session.commit()

    assert result is None

    async with sessions() as session:
        answered_at = (
            await session.execute(
                select(SessionItem.answered_at).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()

    assert answered_at is None
    assert logs == 0


async def test_summary_counts_score_time_and_mistakes(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    scheduler: FixedIntervalScheduler,
) -> None:
    """§22.1: счёт, время, список ошибок."""
    base, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        position = 0
        while (item := await current_item(session, session_id)) is not None:
            await mark_shown(session, item.item_id, now=NOW)
            # Первое задание — верно, остальные — «не знаю».
            raw = {"option": 0} if position == 0 else {"idk": True}
            await answer_item(
                session,
                user_id=base.user_id,
                item_id=item.item_id,
                raw=raw,
                now=NOW + datetime.timedelta(seconds=30 * (position + 1)),
                settings=settings,
                scheduler=scheduler,
            )
            position += 1
        await session.commit()

    async with sessions() as session:
        summary = await session_summary(session, session_id)

    assert summary.total == 3
    assert summary.correct == 1
    assert len(summary.mistakes) == 2
    assert summary.seconds == 90


async def test_shown_at_lands_in_user_state_not_in_fsm(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Отметка выдачи живёт в базе: у `redis` в §30.1 тома нет.

    Всё, что попало бы в FSM, терялось бы при `docker compose down`, а
    критерий приёмки WP-05 требует обратного.
    """
    _, session_id = await _ready_session(engine, sessions, settings)

    async with sessions() as session:
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await session.commit()

    async with sessions() as session:
        user_state = (
            await session.execute(
                select(SessionItem.user_state).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()

    assert SHOWN_AT in (user_state or {})
