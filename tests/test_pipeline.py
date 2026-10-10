"""Машина стадий: атомарность, возобновление, идемпотентность. ТЗ §4.1–§4.3, §4.6.

Атомарность проверяется **по одному тесту на каждую стадию**, а не одним
общим. §4.2 обещает свойство для каждой из шести, и общий тест на одной
оставил бы пять непроверенными.

Это не формальность. Порядок «удалить предыдущие частичные артефакты →
сохранить результат» внутри транзакции легко написать наоборот, и на успешном
пути оба порядка дают одинаковый результат. Разница видна только на сбое между
двумя действиями — то есть ровно в этих тестах.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.base import ExtractedFragment, SourceMeta
from core.adapters.stub import StubAdapter
from core.db.models import Fact, Fragment, Material, Question, ReviewState, Section, User
from core.ingest import handlers as h
from core.ingest.pipeline import StageContext, StageHandler, process_material, run_stage
from core.ingest.stages import ORDER, next_stage, reopen_for_questions
from core.storage import MaterialStorage
from tests.llm_fakes import fake_client
from tests.test_dispatcher import _settings

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _handlers() -> dict[str, StageHandler]:
    """Карта стадий та же, что в работе, — с подделкой провайдера внутри.

    Собирается через `build_handlers`, а не перечислением: иначе стадия,
    добавленная в работе, осталась бы непроверенной здесь, а тесты
    возобновления параметризованы по `ORDER` и прошли бы на недостающем
    обработчике как на пропущенной стадии.
    """
    settings = _settings()
    return h.build_handlers(fake_client(settings), settings)


def _adapter(count: int = 3) -> StubAdapter:
    return StubAdapter(
        meta=SourceMeta(pages=count, text_chars=100 * count),
        fragments=[
            ExtractedFragment(
                ord=i,
                kind="text",
                locator={"type": "web", "url": f"stub://{i}", "quote": f"цитата {i}"},
                text=f"Утверждение номер {i}.",
                quality=1.0,
                tokens=10,
            )
            for i in range(1, count + 1)
        ],
    )


async def _material(engine: AsyncEngine, tg_id: int = 500) -> int:
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
                        title="проба",
                        origin={"sha256": f"hash{tg_id}"},
                        status="processing",
                        mode="full",
                        completed_stages=[],
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


async def _completed(engine: AsyncEngine, material_id: int) -> list[str]:
    async with engine.connect() as connection:
        value = (
            await connection.execute(
                select(Material.completed_stages).where(Material.id == material_id)
            )
        ).scalar_one()
    return list(value or [])


async def _count(engine: AsyncEngine, column: object, where: object) -> int:
    async with engine.connect() as connection:
        return int(
            (
                await connection.execute(select(func.count()).select_from(column).where(where))
            ).scalar_one()
        )


async def _run(
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    material_id: int,
    handlers: dict[str, StageHandler] | None = None,
    adapter: StubAdapter | None = None,
) -> None:
    await process_material(
        sessions,
        storage,
        handlers if handlers is not None else _handlers(),
        material_id,
        adapter if adapter is not None else _adapter(),
        "stub://source",
    )


# --- Полный проход --------------------------------------------------------


async def test_full_run_completes_every_stage(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """`make seed` на заглушках: материал проходит все шесть стадий."""
    material_id = await _material(engine)
    await _run(sessions, MaterialStorage(tmp_path), material_id)

    completed = await _completed(engine, material_id)
    assert completed == [stage.name for stage in ORDER]
    assert next_stage(completed) is None


async def test_full_run_produces_learnable_material(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """После прохода есть фрагменты, факты, вопросы и состояния повторения.

    Последнее — главное: §4.6 требует строку `review_states` для каждого
    обучаемого факта, и без неё планировщик физически не видит материал. Три
    фрагмента дают три факта, по два вопроса на факт.
    """
    material_id = await _material(engine)
    await _run(sessions, MaterialStorage(tmp_path), material_id)

    assert await _count(engine, Fragment, Fragment.material_id == material_id) == 3
    assert await _count(engine, Section, Section.material_id == material_id) == 1
    assert await _count(engine, Fact, Fact.material_id == material_id) == 3

    async with engine.connect() as connection:
        questions = (
            await connection.execute(
                select(func.count())
                .select_from(Question)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id)
            )
        ).scalar_one()
        states = (
            await connection.execute(
                select(func.count())
                .select_from(ReviewState)
                .join(Fact, Fact.id == ReviewState.fact_id)
                .where(Fact.material_id == material_id)
            )
        ).scalar_one()
    assert questions == 6
    assert states == 3, "обучаемые факты не получили review_states — §4.6"


# --- Атомарность: по тесту на каждую стадию (§4.2) -------------------------


@pytest.mark.parametrize("failing", [stage.name for stage in ORDER])
async def test_stage_failure_leaves_nothing_behind(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    failing: str,
) -> None:
    """Сбой стадии не меняет `completed_stages` и не оставляет артефактов.

    Сбой внедряется **после** работы обработчика, то есть после подготовки и
    сохранения результата, но до коммита — самый опасный момент: данные в
    сессии есть, отметки ещё нет. Если транзакционная граница неверна, строки
    останутся, а стадия будет считаться невыполненной, и повторный запуск
    удвоит результат.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    handlers = _handlers()
    original = handlers[failing]

    async def failing_handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        await original(session, material, ctx)
        (ctx.tmp_dir / "artifact.bin").write_bytes(b"x")
        raise RuntimeError(f"сбой в стадии {failing}")

    handlers[failing] = failing_handler

    with pytest.raises(RuntimeError):
        await _run(sessions, storage, material_id, handlers)

    completed = await _completed(engine, material_id)
    assert failing not in completed, f"стадия {failing} отмечена выполненной после сбоя"
    assert completed == [stage.name for stage in ORDER[: ORDER.index(h_stage(failing))]]
    assert not storage.tmp_dir(material_id, failing).exists(), "временные артефакты остались"
    assert not (
        storage.permanent_dir(material_id) / "artifact.bin"
    ).exists(), "артефакт сбойной стадии попал в постоянное хранилище"


def h_stage(name: str) -> object:
    from core.ingest.stages import BY_NAME

    return BY_NAME[name]


@pytest.mark.parametrize("failing", [stage.name for stage in ORDER])
async def test_pipeline_resumes_after_failure(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    failing: str,
) -> None:
    """После сбоя повторный запуск доводит материал до конца.

    §4.3: воркер начинает с первой незавершённой стадии. Проверяется для
    каждой — иначе одна выпавшая стадия давала бы материал, застрявший без
    видимой причины.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    handlers = _handlers()
    original = handlers[failing]
    attempts = {"n": 0}

    async def flaky(session: AsyncSession, material: object, ctx: StageContext) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("первая попытка падает")
        await original(session, material, ctx)

    handlers[failing] = flaky

    with pytest.raises(RuntimeError):
        await _run(sessions, storage, material_id, handlers)
    await _run(sessions, storage, material_id, handlers)

    assert await _completed(engine, material_id) == [stage.name for stage in ORDER]


async def test_repeated_stage_does_not_duplicate_results(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повторный проход по готовому материалу — no-op (§4.3).

    Ассерт на число строк, а не на отсутствие ошибки: удвоение фрагментов
    прошло бы без исключения и обнаружилось бы только в конспекте.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    await _run(sessions, storage, material_id)
    await _run(sessions, storage, material_id)

    assert await _count(engine, Fragment, Fragment.material_id == material_id) == 3
    assert await _count(engine, Fact, Fact.material_id == material_id) == 3


async def test_completed_stage_does_not_call_handler_again(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Готовая стадия не вызывает обработчик вовсе.

    Положительный ассерт к предыдущему тесту: одинаковое число строк вышло бы
    и у идемпотентного обработчика, вызванного дважды. Здесь проверяется, что
    он не вызван.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    adapter = _adapter()
    await _run(sessions, storage, material_id, adapter=adapter)
    assert adapter.extract_calls == 1

    await _run(sessions, storage, material_id, adapter=adapter)
    assert adapter.extract_calls == 1, "стадия extract выполнена повторно"


async def test_run_stage_directly_skips_completed_stage(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """`run_stage` на готовой стадии возвращает `False` и не зовёт обработчик.

    Отдельный тест, потому что через `process_material` эта проверка
    недостижима: внешний цикл идёт по `next_stage` и готовую стадию не
    выбирает вовсе. Мутационный прогон это и показал — снятие проверки
    внутри `run_stage` не уронило ни одного теста.

    Путь реальный: кнопка «Достроить задания» (§6.3) и повторная постановка
    задачи ARQ после сбоя вызывают стадию адресно, а не через весь конвейер.
    Без проверки повторный вызов удвоил бы результат стадии.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    await _run(sessions, storage, material_id)

    called = {"n": 0}

    async def counting(session: AsyncSession, material: object, ctx: StageContext) -> None:
        called["n"] += 1

    from core.ingest.stages import EXTRACT

    def ctx_factory(tmp_dir: Path) -> StageContext:
        return StageContext(
            material_id=material_id,
            tmp_dir=tmp_dir,
            adapter=_adapter(),
            source_path="stub://source",
        )

    result = await run_stage(sessions, storage, counting, EXTRACT, ctx_factory, material_id)

    assert result is False, "готовая стадия объявлена выполненной заново"
    assert called["n"] == 0, "обработчик готовой стадии был вызван"
    assert not storage.tmp_dir(
        material_id, "extract"
    ).exists(), "временная директория создана для стадии, которую не выполняли"


# --- §4.6 и §38 №16, №20 --------------------------------------------------


async def test_schedule_does_not_reset_accumulated_state(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повторный `schedule` не сбрасывает накопленное повторение. §38 №16.

    Следствие И-7: идентичность факта стабильна, и расписание не обнуляется
    при переобработке материала. Проверяется через `reps`, выставленный
    вручную: если вставка окажется не идемпотентной, он вернётся в ноль.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    await _run(sessions, storage, material_id)

    async with engine.begin() as connection:
        await connection.execute(ReviewState.__table__.update().values(reps=7, seen=True))

    completed = await _completed(engine, material_id)
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=[s for s in completed if s != "schedule"])
        )
    await _run(sessions, storage, material_id)

    async with engine.connect() as connection:
        reps = list((await connection.execute(select(ReviewState.reps))).scalars())
    assert reps == [7, 7, 7], f"состояние повторения сброшено: {reps}"


async def test_restarted_questions_stage_drops_drafts(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Перезапуск `questions` удаляет `draft`-строки. §38 №20, §4.2.

    Это вторая половина защиты: в WP-02 проверялось, что частичный уникальный
    индекс допускает `draft` рядом с `valid`, здесь — что код действительно
    чистит. Без этого упавший воркер оставляет занятые слоты
    `(fact_id, type, direction)` навсегда, и факт теряет формат вопроса.
    """
    material_id = await _material(engine)
    storage = MaterialStorage(tmp_path)
    handlers = _handlers()

    async def dies_after_draft(session: AsyncSession, material: object, ctx: StageContext) -> None:
        fact_id = (
            await session.execute(
                select(Fact.id).where(Fact.material_id == ctx.material_id).limit(1)
            )
        ).scalar_one()
        session.add(
            Question(
                fact_id=fact_id,
                type="mcq",
                direction="forward",
                payload={},
                answer={},
                difficulty=2,
                status="draft",
                created_at=NOW,
            )
        )
        await session.flush()

    handlers["questions"] = dies_after_draft
    await _run(sessions, storage, material_id, handlers)

    async with engine.connect() as connection:
        drafts = (
            await connection.execute(
                select(func.count())
                .select_from(Question)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id, Question.status == "draft")
            )
        ).scalar_one()
    assert drafts == 1, "подготовка теста не создала draft"

    # Перезапуск стадии: снимаем отметки и прогоняем настоящий обработчик.
    completed = await _completed(engine, material_id)
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=reopen_for_questions(completed))
        )
    await _run(sessions, storage, material_id)

    async with engine.connect() as connection:
        remaining = (
            await connection.execute(
                select(func.count())
                .select_from(Question)
                .join(Fact, Fact.id == Question.fact_id)
                .where(Fact.material_id == material_id, Question.status == "draft")
            )
        ).scalar_one()
    assert remaining == 0, "draft-строки пережили перезапуск стадии — §38 №20"


async def test_extract_rejects_fragment_without_locator(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Фрагмент без локатора отбрасывает всю стадию.

    §3 объявляет `locator` обязательным, и без него фрагмент не даёт перехода
    к источнику — то есть И-1 нарушается не на уровне факта, а раньше.
    Проверяется, что стадия падает целиком, а не сохраняет остальные
    фрагменты: половина материала хуже, чем отсутствие материала, потому что
    выглядит рабочей.
    """
    material_id = await _material(engine)
    broken = StubAdapter(
        fragments=[
            ExtractedFragment(ord=1, kind="text", locator={"type": "web"}, text="хороший"),
            ExtractedFragment(ord=2, kind="text", locator={}, text="без локатора"),
        ]
    )

    with pytest.raises(ValueError, match="без локатора"):
        await _run(sessions, MaterialStorage(tmp_path), material_id, adapter=broken)

    assert await _count(engine, Fragment, Fragment.material_id == material_id) == 0
    assert await _completed(engine, material_id) == []
