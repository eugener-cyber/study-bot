"""Сквозной путь: конвейер → расписание → сеанс → ответ. Свойства §38 №15 и №16.

Здесь проверяются критерии приёмки WP-05, которые нельзя закрыть юнит-тестом:
они про стык конвейера обработки материала и педагогического слоя.

Конвейер запускается **настоящий** — через `process_material` с подделкой
только на месте провайдера LLM. Помощники взяты из `tests/test_pipeline.py`, а
не переписаны: карта стадий там собирается через `build_handlers`, и своя копия
разошлась бы с рабочей при первой же новой стадии.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.db.engine import build_engine, build_sessionmaker
from core.db.models import Fact, Question, ReviewLog, ReviewState, Session, SessionItem
from core.db.teachable import teachable_fact_ids
from core.ingest.handlers import ensure_review_states
from core.ingest.stages import reopen_for_questions
from core.learning.scheduler import FixedIntervalScheduler
from core.services.sessions import (
    SHOWN_AT,
    active_session_id,
    answer_item,
    assemble,
    current_item,
    mark_shown,
)
from core.storage import MaterialStorage
from tests.conftest import _test_url
from tests.test_dispatcher import _settings
from tests.test_pipeline import _adapter, _completed, _material, _run

pytestmark = pytest.mark.db

NOW = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
"""Опорный момент — **относительно настоящих часов**, а не календарная дата.

Фиксированная дата здесь была дефектом: стадия `schedule` ставит
`due_at = processed_at + NEW_FACT_DELAY` по `datetime.now` (§4.6,
`core/ingest/handlers.py`), и при `NOW` в прошлом ни один факт не наступал —
тесты были зелёными только в день, когда их написали, и красными на
следующий. Час вперёд берётся с запасом к `NEW_FACT_DELAY_MIN = 10`.

Остальные тесты пакета фиксированный момент сохраняют и правильно: там срок
задаёт фабрика, а не настоящие часы.
"""

LATER = NOW + datetime.timedelta(seconds=10)


@pytest.fixture
def storage(tmp_path: object) -> MaterialStorage:
    return MaterialStorage(str(tmp_path))


async def _processed_material(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    *,
    tg_id: int = 700,
    fragments: int = 3,
) -> tuple[int, int]:
    """Материал, прогнанный через конвейер целиком. Возвращает `(user_id, material_id)`.

    Подделка провайдера даёт один факт на фрагмент, поэтому число фрагментов —
    это число обучаемых фактов, и тесту, которому нужен запас сверх
    `MIN_SESSION_ITEMS`, достаточно попросить больше.
    """
    material_id = await _material(engine, tg_id=tg_id)
    await _run(sessions, storage, material_id, adapter=_adapter(fragments))

    async with engine.connect() as connection:
        user_id = (
            (
                await connection.execute(
                    select(ReviewState.user_id).where(
                        ReviewState.fact_id.in_(
                            select(Fact.id).where(Fact.material_id == material_id)
                        )
                    )
                )
            )
            .scalars()
            .first()
        )
    assert user_id is not None, "конвейер не создал ни одной строки расписания"
    return int(user_id), material_id


async def _snapshot(engine: AsyncEngine, user_id: int) -> list[tuple[object, ...]]:
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                select(
                    ReviewState.fact_id,
                    ReviewState.due_at,
                    ReviewState.reps,
                    ReviewState.lapses,
                    ReviewState.seen,
                    ReviewState.introduced_at,
                    ReviewState.last_review,
                )
                .where(ReviewState.user_id == user_id)
                .order_by(ReviewState.fact_id)
            )
        ).all()
    return [tuple(row) for row in rows]


async def test_every_teachable_fact_has_a_review_state_and_the_first_session_is_reachable(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    storage: MaterialStorage,
) -> None:
    """**§38 №15.** Доказывается достижением сеанса, а не наличием строк.

    Фикстура намеренно содержит факт, у которого валиден только `open`: он
    обучаем по §14.3 — валидный вопрос есть, — получает строку расписания, но
    задания в WP-05 не даёт (Issue #31). Так тест измеряет разрыв, а не
    обходит его.
    """
    settings = _settings()
    # Четыре факта, а не три: один из них лишится mcq, и трёх оставшихся
    # ровно хватает на `MIN_SESSION_ITEMS` — иначе тест проверял бы порог §16,
    # а не свойство №15.
    user_id, material_id = await _processed_material(engine, sessions, storage, fragments=4)

    # Факт, у которого mcq отклонён: остаётся обучаемым по open-вопросу.
    async with engine.begin() as connection:
        open_only_fact = (
            await connection.execute(
                select(Question.fact_id)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id, Question.type == "open")
                .limit(1)
            )
        ).scalar_one()
        await connection.execute(
            Question.__table__.update()
            .where(Question.fact_id == open_only_fact, Question.type == "mcq")
            .values(status="rejected", reject_reason="тест §38 №15: остался только open")
        )

    async with sessions() as session:
        teachable = {
            int(row)
            for row in (
                await session.execute(teachable_fact_ids(settings.MIN_FACT_CONFIDENCE))
            ).scalars()
        }
        scheduled = {
            int(row)
            for row in (
                await session.execute(
                    select(ReviewState.fact_id).where(ReviewState.user_id == user_id)
                )
            ).scalars()
        }

    assert teachable, "обучаемых фактов нет — проверять нечего"
    assert open_only_fact in teachable, "факт с одним open перестал быть обучаемым"
    assert teachable == scheduled, "у обучаемого факта нет строки review_states"

    async with sessions() as session:
        session_id = await assemble(session, user_id=user_id, settings=settings, now=NOW)
        await session.commit()
    assert session_id is not None, "первая сессия недостижима — §38 №15 не выполнен"

    async with sessions() as session:
        offered = {
            int(row)
            for row in (
                await session.execute(
                    select(SessionItem.fact_id).where(SessionItem.session_id == session_id)
                )
            ).scalars()
        }
    assert open_only_fact not in offered, "выдано задание, на которое нечем ответить"


async def test_rerunning_the_schedule_stage_keeps_accumulated_state(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    storage: MaterialStorage,
) -> None:
    """**§38 №16.** Повторный запуск стадии не сбрасывает накопленное.

    Проверяется после ответов, а не на свежих строках: до первого ответа
    `reps`, `seen` и `introduced_at` пусты, и сброс был бы неотличим от
    исходного состояния — тест остался бы зелёным при `ON CONFLICT DO UPDATE`.
    """
    settings = _settings()
    user_id, material_id = await _processed_material(engine, sessions, storage)

    async with sessions() as session:
        session_id = await assemble(session, user_id=user_id, settings=settings, now=NOW)
        assert session_id is not None
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        answered = await answer_item(
            session,
            user_id=user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=FixedIntervalScheduler(),
        )
        await session.commit()
    assert answered is not None, "ответ не прошёл — сбрасывать было бы нечего"

    before = await _snapshot(engine, user_id)
    assert any(row[2] for row in before), "ни у одного факта нет ответов"

    async with sessions() as session:
        created = await ensure_review_states(session, material_id, settings)
        await session.commit()

    after = await _snapshot(engine, user_id)
    assert created == 0, "повторный запуск создал строки заново"
    assert after == before, "повторный запуск стадии schedule сбросил состояние"


async def test_unfinished_session_survives_a_fresh_connection(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    storage: MaterialStorage,
) -> None:
    """Состояние сеанса живёт в базе, а не в памяти процесса или в FSM.

    Это автоматизируемая половина критерия «после `docker compose down && up`
    состояние не теряется»: у `redis` в §30.1 тома нет, поэтому всё, что
    попало бы в FSM, пропало бы. Новый движок и новый пул соединений — то, что
    остаётся от перезапуска процесса в пределах теста; живой `down && up`
    проверяется руками, и его вывод приложен к PR.
    """
    settings = _settings()
    user_id, _ = await _processed_material(engine, sessions, storage)

    async with sessions() as session:
        session_id = await assemble(session, user_id=user_id, settings=settings, now=NOW)
        assert session_id is not None
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        await session.commit()

    await engine.dispose()

    fresh_engine = build_engine(_test_url())
    fresh_sessions = build_sessionmaker(fresh_engine)
    try:
        async with fresh_sessions() as session:
            resumed = await active_session_id(session, user_id)
            assert resumed == session_id, "активный сеанс не найден после переподключения"

            resumed_item = await current_item(session, resumed)
            assert resumed_item is not None
            assert resumed_item.item_id == item.item_id
            assert resumed_item.position == 1

            user_state = (
                await session.execute(
                    select(SessionItem.user_state).where(SessionItem.id == item.item_id)
                )
            ).scalar_one()
            assert SHOWN_AT in (user_state or {}), "отметка выдачи не пережила переподключение"
    finally:
        await fresh_engine.dispose()


async def test_rerunning_the_questions_stage_on_answered_questions_loses_nothing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    storage: MaterialStorage,
) -> None:
    """Issue #28: удаление валидных вопросов станет опасным — но не сегодня.

    Состояние для теста собрано **искусственно**: `completed_stages` правится
    прямо здесь, потому что рабочего пути к повторному прогону стадии
    `questions` не существует ни сейчас, ни после WP-05 — кнопки «Достроить
    задания» нет ни в одном хендлере (Issue #32).

    Тест фиксирует, что будет, когда путь появится: стадия падает на внешнем
    ключе `review_logs.question_id` и по §4.2 не оставляет частичного
    результата — сеанс, журнал и ответы целы. Исправление — WP-10 вместе со
    статусом `draft` (§11.5).
    """
    settings = _settings()
    user_id, material_id = await _processed_material(engine, sessions, storage)

    async with sessions() as session:
        session_id = await assemble(session, user_id=user_id, settings=settings, now=NOW)
        assert session_id is not None
        item = await current_item(session, session_id)
        assert item is not None
        await mark_shown(session, item.item_id, now=NOW)
        answered = await answer_item(
            session,
            user_id=user_id,
            item_id=item.item_id,
            raw={"option": 0},
            now=LATER,
            settings=settings,
            scheduler=FixedIntervalScheduler(),
        )
        await session.commit()
    assert answered is not None

    async with engine.connect() as connection:
        logs_before = (
            await connection.execute(select(func.count()).select_from(ReviewLog))
        ).scalar_one()
        questions_before = (
            await connection.execute(
                select(func.count())
                .select_from(Question)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id)
            )
        ).scalar_one()
    assert logs_before == 1

    completed = await _completed(engine, material_id)
    async with engine.begin() as connection:
        await connection.execute(
            Fact.__table__.metadata.tables["materials"]
            .update()
            .where(Fact.__table__.metadata.tables["materials"].c.id == material_id)
            .values(completed_stages=reopen_for_questions(completed))
        )

    with pytest.raises(Exception):  # noqa: B017 — класс исключения задаёт драйвер
        await _run(sessions, storage, material_id)

    async with engine.connect() as connection:
        logs_after = (
            await connection.execute(select(func.count()).select_from(ReviewLog))
        ).scalar_one()
        questions_after = (
            await connection.execute(
                select(func.count())
                .select_from(Question)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id)
            )
        ).scalar_one()
        answered_at = (
            await connection.execute(
                select(SessionItem.answered_at).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()
        status = (
            await connection.execute(select(Session.status).where(Session.id == session_id))
        ).scalar_one()

    assert logs_after == logs_before, "журнал ответов пострадал"
    assert questions_after == questions_before, "вопросы удалены частично"
    assert answered_at is not None, "ответ потерян"
    assert status == "active"
