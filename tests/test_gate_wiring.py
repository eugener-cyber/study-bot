"""Гейт бюджета в конвейере и режим `notes_only`. ТЗ §6.3, §4.1, §4.5.

`test_budget.py` проверяет арифметику, здесь — что гейт **подключён** и
останавливает обработку в нужный момент. Разница существенная: правильный
расчёт, не вызванный из конвейера, прошёл бы все тесты первого файла.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.base import ExtractedFragment, SourceMeta
from core.adapters.stub import StubAdapter
from core.db.models import Fact, Material, Question, ReviewState, User
from core.db.session import session_scope
from core.ingest.awaiting import AWAITING_STAGE, is_awaiting
from core.ingest.gate import choose_full, choose_notes_only, probe_and_gate
from core.ingest.pipeline import AwaitingUser, process_material
from core.ingest.stages import ORDER, reopen_for_questions
from core.storage import MaterialStorage
from tests.test_dispatcher import _settings
from tests.test_pipeline import _handlers

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _material(engine: AsyncEngine, tg_id: int = 950) -> int:
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


def _adapter(count: int = 2, *, pages: int = 1, vision: float = 0.0) -> StubAdapter:
    return StubAdapter(
        meta=SourceMeta(pages=pages, vision_pages_ratio=vision),
        fragments=[
            ExtractedFragment(
                ord=i,
                kind="text",
                locator={"type": "web", "url": f"stub://{i}"},
                text=f"Утверждение {i}.",
                tokens=100,
            )
            for i in range(1, count + 1)
        ],
    )


# --- Гейт срабатывает ДО extract ------------------------------------------


async def test_gate_stops_before_extract(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Главный критерий: `extract` не вызывается, если гейт сработал.

    Ассерт именно на счётчик вызовов адаптера, а не на «гейт сработал».
    Вторая формулировка прошла бы и при гейте, поставленном после `extract`, —
    то есть при том дефекте, из-за которого §6.3 переписывался в v3.5.
    """
    settings = _settings(COST_WARN_THRESHOLD="0.01")
    material_id = await _material(engine)
    adapter = _adapter(pages=500, vision=1.0)

    with pytest.raises(AwaitingUser):
        await probe_and_gate(sessions, material_id, adapter, "stub://x", settings)

    assert adapter.probe_calls == 1
    assert adapter.extract_calls == 0, "extract вызван при сработавшем гейте"


async def test_gate_parks_material_with_a_question(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Материал остаётся в `processing` и ждёт ответа (§4.5).

    Не `failed`: отказа не было, пользователя спросили. Вопрос содержит
    цифры — без них выбор между «обработать полностью» и «только конспект»
    делается наугад (§6.3).
    """
    settings = _settings(COST_WARN_THRESHOLD="0.01")
    material_id = await _material(engine)

    with pytest.raises(AwaitingUser) as raised:
        await probe_and_gate(
            sessions, material_id, _adapter(pages=500, vision=1.0), "stub://x", settings
        )

    assert "≈" in str(raised.value)
    async with session_scope(sessions) as session:
        assert await is_awaiting(session, material_id) is True

    async with engine.connect() as connection:
        status, progress, estimate = (
            await connection.execute(
                select(Material.status, Material.progress, Material.cost_estimate).where(
                    Material.id == material_id
                )
            )
        ).one()
    assert status == "processing"
    assert progress["stage"] == AWAITING_STAGE
    assert progress["kind"] == "budget"
    assert estimate is not None, "оценка не сохранена в cost_estimate — §6.3"


async def test_estimate_is_saved_even_when_gate_does_not_fire(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Оценка пишется всегда, а не только при срабатывании.

    §6.4 сравнивает оценку с фактическим расходом, и материал без оценки
    выпал бы из этого сравнения незаметно.
    """
    settings = _settings()
    material_id = await _material(engine)

    await probe_and_gate(sessions, material_id, _adapter(), "stub://x", settings)

    async with engine.connect() as connection:
        estimate = (
            await connection.execute(
                select(Material.cost_estimate).where(Material.id == material_id)
            )
        ).scalar_one()
    assert estimate is not None
    assert "est_cost" in estimate
    assert "currency" in estimate


async def test_gate_does_not_fire_without_threshold(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """При незаданном `COST_WARN_THRESHOLD` обработка идёт.

    Порог в §30.3 стоит как `TODO(owner)` и выводится из спайка, который не
    выполнялся. Срабатывание на `None` остановило бы любой материал — продукт
    не работал бы вовсе до получения ключа провайдера.
    """
    settings = _settings()
    assert settings.COST_WARN_THRESHOLD is None
    material_id = await _material(engine)

    estimate = await probe_and_gate(
        sessions, material_id, _adapter(pages=10_000, vision=1.0), "stub://x", settings
    )

    assert estimate.est_cost > 0
    async with session_scope(sessions) as session:
        assert await is_awaiting(session, material_id) is False


# --- Режим notes_only (§6.3) ----------------------------------------------


async def test_notes_only_skips_questions_and_reaches_ready(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§6.3: `questions:skipped` в `completed_stages`, `schedule` выполняется.

    И ни одной строки `review_states`: у фактов нет валидных вопросов, значит
    они не обучаемы (§14.3). Отдельного условия для этого нет — работает
    предикат из WP-02, и это проверяется здесь, а не предполагается.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        await choose_notes_only(session, material_id)

    await process_material(
        sessions, MaterialStorage(tmp_path), _handlers(), material_id, _adapter(), "stub://x"
    )

    async with engine.connect() as connection:
        completed = list(
            (
                await connection.execute(
                    select(Material.completed_stages).where(Material.id == material_id)
                )
            ).scalar_one()
        )
        questions = (await connection.execute(select(Question.id))).all()
        states = (await connection.execute(select(ReviewState.fact_id))).all()
        facts = (await connection.execute(select(Fact.id))).all()

    assert "questions:skipped" in completed, "пропуск не отмечен отдельным маркером"
    assert "questions" not in completed, "пропуск записан как выполнение"
    assert "schedule" in completed, "schedule не выполнена — §6.3 требует выполнить"
    assert len(facts) == 2, "факты не созданы"
    assert questions == [], "в режиме notes_only созданы вопросы"
    assert states == [], "необучаемые факты получили review_states"


async def test_build_questions_button_completes_the_material(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Кнопка «Достроить задания» доводит материал до полного (§6.3).

    Проверяется и то, что ранние стадии **не** переисполнялись: §6.3 требует
    запускать конвейер со стадии `questions`, а повторное извлечение для
    сканированного PDF означало бы повторную оплату самой дорогой стадии.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    adapter = _adapter()

    async with session_scope(sessions) as session:
        await choose_notes_only(session, material_id)
    await process_material(sessions, storage, _handlers(), material_id, adapter, "stub://x")
    extract_calls_before = adapter.extract_calls

    # Нажатие кнопки: режим в `full`, отметки `questions:skipped` и `schedule`
    # сняты, дальше обычное возобновление §4.3.
    async with session_scope(sessions) as session:
        await choose_full(session, material_id)
    async with engine.connect() as connection:
        completed = list(
            (
                await connection.execute(
                    select(Material.completed_stages).where(Material.id == material_id)
                )
            ).scalar_one()
        )
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=reopen_for_questions(completed))
        )

    await process_material(sessions, storage, _handlers(), material_id, adapter, "stub://x")

    async with engine.connect() as connection:
        final = list(
            (
                await connection.execute(
                    select(Material.completed_stages).where(Material.id == material_id)
                )
            ).scalar_one()
        )
        questions = (await connection.execute(select(Question.id))).all()
        states = (await connection.execute(select(ReviewState.fact_id))).all()

    assert final == [stage.name for stage in ORDER], f"итоговые стадии: {final}"
    assert len(questions) == 4, "задания не достроены"
    assert len(states) == 2, "обучаемые факты не получили review_states"
    assert adapter.extract_calls == extract_calls_before, "extract переисполнен"
