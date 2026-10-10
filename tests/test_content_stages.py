"""Содержательные стадии на живой базе. ТЗ §7.1, §7.3, §8, §10.1, §6.4.

`test_llm_client.py` проверяет слой, здесь — что стадии **им пользуются** и
правильно распоряжаются ответом модели. Разница та же, что между `test_budget`
и `test_gate_wiring`: верный расчёт, не вызванный из конвейера, прошёл бы все
тесты первого файла.

Ответы модели подделываются по назначению вызова, а не строятся из промпта
там, где проверяется реакция на **конкретный** ответ: выдуманный
идентификатор фрагмента, повтор слота, факт-вывод. Построенный из промпта
ответ всегда корректен, и эти пути остались бы непроверенными.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.base import ExtractedFragment, SourceMeta
from core.adapters.stub import StubAdapter
from core.db.models import Fact, Fragment, LlmCall, Material, Question, Section, User
from core.db.session import session_scope
from core.ingest import handlers as h
from core.ingest.handlers import UNASSIGNED_TITLE, WARNING_MARK, norm_hash, render_note
from core.ingest.pipeline import StageContext
from core.llm.client import DatabaseRecorder
from core.llm.schemas import SectionsAnswer
from core.storage import MaterialStorage
from tests.llm_fakes import ScriptedProvider, fake_client
from tests.test_dispatcher import _settings

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _adapter(count: int = 3) -> StubAdapter:
    return StubAdapter(
        meta=SourceMeta(pages=count, text_chars=100 * count),
        fragments=[
            ExtractedFragment(
                ord=index,
                kind="text",
                locator={"type": "web", "url": f"stub://{index}", "quote": f"цитата {index}"},
                text=f"Утверждение номер {index}.",
                quality=1.0,
                tokens=10,
            )
            for index in range(1, count + 1)
        ],
    )


async def _material(engine: AsyncEngine, tg_id: int = 900) -> int:
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


async def _run_stages(
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    material_id: int,
    stages: tuple[str, ...],
    *,
    provider: ScriptedProvider | None = None,
    recorder: object | None = None,
) -> ScriptedProvider:
    """Вызывает обработчики стадий напрямую, в транзакции на стадию.

    Не через `run_stage`: тот по §4.3 считает уже выполненную стадию no-op, а
    здесь стадия нарочно прогоняется второй раз с другим ответом модели —
    проверяется её собственное поведение, а не машина стадий. Машину
    проверяет `test_pipeline.py`, и дублировать её здесь значило бы проверять
    дважды одно и то же, а второй раз — хуже.
    """
    settings = _settings()
    provider = provider or ScriptedProvider()
    client = fake_client(
        settings,
        provider=provider,
        recorder=recorder,  # type: ignore[arg-type]
    )
    handlers = h.build_handlers(client, settings)
    adapter = _adapter()

    for stage in stages:
        ctx = StageContext(
            material_id=material_id,
            tmp_dir=storage.prepare_tmp(material_id, stage),
            adapter=adapter,
            source_path="/tmp/none.md",
        )
        async with session_scope(sessions) as session:
            material = (
                await session.execute(select(Material).where(Material.id == material_id))
            ).scalar_one()
            await handlers[stage](session, material, ctx)
    return provider


async def _rows(engine: AsyncEngine, statement: object) -> list[tuple[object, ...]]:
    async with engine.connect() as connection:
        return [tuple(row) for row in (await connection.execute(statement)).all()]  # type: ignore[arg-type]


# --- sections -------------------------------------------------------------


async def test_sections_keep_every_fragment(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Все фрагменты распределены, порядковые номера без дыр.

    `ord` без дыр существенен: по нему идёт навигация §8.1, и пропуск
    означал бы страницу, на которую нельзя перейти.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections"))

    rows = await _rows(
        engine,
        select(Section.ord, Section.fragment_ids)
        .where(Section.material_id == material_id)
        .order_by(Section.ord),
    )
    assigned = [fragment_id for _, ids in rows for fragment_id in ids]
    assert [ord_ for ord_, _ in rows] == list(range(1, len(rows) + 1))
    assert len(assigned) == len(set(assigned)) == 3


async def test_invented_fragment_id_is_dropped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Выдуманный идентификатор отбрасывается — §7.3 проверка 1 по смыслу.

    Без этого ссылки фиктивны: секция ссылалась бы на фрагмент, которого в
    материале нет, и `locator` §3 не разрешился бы ни в какую цитату.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract",))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={
            "sections": {
                "sections": [
                    {"title": "С выдумкой", "fragment_ids": [real[0], 999_999]},
                    {"title": "Остальное", "fragment_ids": real[1:]},
                ]
            }
        }
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("sections",), provider=provider
    )

    rows = await _rows(
        engine, select(Section.fragment_ids).where(Section.material_id == material_id)
    )
    assigned = [fragment_id for (ids,) in rows for fragment_id in ids]  # type: ignore[union-attr]
    assert 999_999 not in assigned
    assert sorted(assigned) == sorted(real)


async def test_fragment_claimed_twice_stays_in_the_first_section(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повтор не удваивает фрагмент.

    Иначе один и тот же текст породил бы два набора фактов, и дедуп §7.4
    получил бы работу, которой могло не быть.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract",))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={
            "sections": {
                "sections": [
                    {"title": "Первая", "fragment_ids": real},
                    {"title": "Вторая", "fragment_ids": real},
                ]
            }
        }
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("sections",), provider=provider
    )

    rows = await _rows(
        engine,
        select(Section.title, Section.fragment_ids)
        .where(Section.material_id == material_id)
        .order_by(Section.ord),
    )
    assert [title for title, _ in rows] == ["Первая"]
    assert sorted(rows[0][1]) == sorted(real)  # type: ignore[arg-type]


async def test_unassigned_fragments_get_their_own_section(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Потерянный фрагмент не исчезает молча.

    Из двух реакций — уронить материал или собрать остаток — выбрана вторая:
    упавший материал виден, но пользователь теряет всё.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract",))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={"sections": {"sections": [{"title": "Только первый", "fragment_ids": real[:1]}]}}
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("sections",), provider=provider
    )

    rows = await _rows(
        engine,
        select(Section.title, Section.fragment_ids)
        .where(Section.material_id == material_id)
        .order_by(Section.ord),
    )
    assert [title for title, _ in rows] == ["Только первый", UNASSIGNED_TITLE]
    assert sorted(rows[1][1]) == sorted(real[1:])  # type: ignore[arg-type]


async def test_answer_made_only_of_invented_ids_keeps_the_material(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Ответ целиком из выдумки не теряет материал, а собирает его в одну секцию.

    Это худший случай разбора ответа, и выбор здесь сознательный: падение
    стадии означало бы потерю материала из-за ошибки модели в нумерации, а
    одна секция `UNASSIGNED_TITLE` сохраняет содержание — факты по нему
    извлекутся. Перекос виден в журнале двумя предупреждениями подряд:
    `sections_unknown_fragments` и `sections_unassigned`.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract",))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={"sections": {"sections": [{"title": "Выдумка", "fragment_ids": [777_777]}]}}
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("sections",), provider=provider
    )

    rows = await _rows(
        engine,
        select(Section.title, Section.fragment_ids).where(Section.material_id == material_id),
    )
    assert [title for title, _ in rows] == [UNASSIGNED_TITLE]
    assert sorted(rows[0][1]) == sorted(real)  # type: ignore[arg-type]


def test_assigning_without_fragments_is_rejected() -> None:
    """Предусловие `_assign_fragments`: пустой список фрагментов — ошибка.

    Достижимо только при неверном вызове, и проверка оставлена именно как
    предусловие: иначе функция вернула бы пустой список секций, и стадия
    записала бы материал без единой секции, не сообщив ничего.
    """
    with pytest.raises(ValueError, match="пуст"):
        h._assign_fragments([], [], material_id=1)


# --- facts ----------------------------------------------------------------


async def test_facts_fill_the_section_summary(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """`summary_md` приходит из вызова §7.1 — поэтому конспект не требует вызова.

    Если бы стадия фактов его не записывала, §8 остался бы без своего входа, и
    конспект начинался бы пустой строкой под заголовком темы.
    """
    material_id = await _material(engine)
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections", "facts")
    )

    rows = await _rows(
        engine,
        select(Section.title, Section.summary_md).where(Section.material_id == material_id),
    )
    assert rows[0][1]
    assert str(rows[0][1]).strip() != ""


async def test_fact_without_valid_evidence_is_dropped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§7.3 проверка 1: факт, ссылающийся в пустоту, отбрасывается.

    Это защита И-1. Вместе с ним проверяется, что факт с **частично** верными
    ссылками сохраняется с их пересечением, а не отбрасывается целиком: §7.3
    требует подмножества, и верная половина evidence остаётся верной.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections"))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={
            "facts": {
                "title": "Тема",
                "summary_md": "Описание.",
                "facts": [
                    {
                        "statement": "Факт без evidence",
                        "detail": "Раскрытие.",
                        "kind": "definition",
                        "fragment_ids": [888_888],
                        "importance": 3,
                        "difficulty": 2,
                        "derived": False,
                        "confidence": 0.9,
                    },
                    {
                        "statement": "Факт с половиной evidence",
                        "detail": "Раскрытие.",
                        "kind": "definition",
                        "fragment_ids": [real[0], 888_888],
                        "importance": 3,
                        "difficulty": 2,
                        "derived": False,
                        "confidence": 0.9,
                    },
                ],
            }
        }
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("facts",), provider=provider
    )

    rows = await _rows(
        engine,
        select(Fact.statement, Fact.fragment_ids).where(Fact.material_id == material_id),
    )
    assert [statement for statement, _ in rows] == ["Факт с половиной evidence"]
    assert rows[0][1] == [real[0]]


async def test_facts_stage_fails_when_nothing_is_learnable(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Материал без фактов не доходит до `ready` молча.

    Иначе пользователь получил бы «готово» и пустой конспект, а причину
    искать было бы негде.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections"))

    provider = ScriptedProvider(
        answers={"facts": {"title": "Титул", "summary_md": "Только оглавление.", "facts": []}}
    )
    with pytest.raises(ValueError, match="нечему учиться"):
        await _run_stages(
            sessions, MaterialStorage(tmp_path), material_id, ("facts",), provider=provider
        )


def test_norm_hash_follows_7_4_normalization() -> None:
    """§7.4: NFKC, lowercase, без пунктуации, схлопнутые пробелы.

    Признак первичный, но не декоративный: по нему §7.4 ищет кандидатов на
    дедуп, и случайный хеш сделал бы будущий дедуп невозможным.
    """
    assert norm_hash("Митоз — деление клетки.") == norm_hash("митоз  деление   клетки")
    assert norm_hash("Митоз") != norm_hash("Мейоз")


# --- questions ------------------------------------------------------------


async def test_questions_are_generated_only_for_eligible_facts(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§7.3 проверки 3 и 4: ни выведенный факт, ни факт ниже порога вопросов не получают.

    Проверяется на трёх фактах сразу — годном, выведенном и с низкой
    уверенностью: проверка на одном негодном прошла бы и при реализации,
    которая не генерирует вопросы вовсе.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections"))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    def draft(statement: str, **overrides: object) -> dict[str, object]:
        return {
            "statement": statement,
            "detail": "Раскрытие.",
            "kind": "definition",
            "fragment_ids": [real[0]],
            "importance": 3,
            "difficulty": 2,
            "derived": False,
            "confidence": 0.9,
            **overrides,
        }

    provider = ScriptedProvider(
        answers={
            "facts": {
                "title": "Тема",
                "summary_md": "Описание.",
                "facts": [
                    draft("Годный факт"),
                    draft("Вывод модели", derived=True),
                    draft("Слабая уверенность", confidence=0.1),
                ],
            }
        }
    )
    await _run_stages(
        sessions,
        MaterialStorage(tmp_path),
        material_id,
        ("facts", "questions"),
        provider=provider,
    )

    rows = await _rows(
        engine,
        select(Fact.statement, func.count(Question.id))
        .join(Question, Question.fact_id == Fact.id, isouter=True)
        .where(Fact.material_id == material_id)
        .group_by(Fact.statement),
    )
    counts = {str(statement): int(count) for statement, count in rows}
    assert counts["Годный факт"] == 2
    assert counts["Вывод модели"] == 0
    assert counts["Слабая уверенность"] == 0


async def test_duplicate_slot_in_the_answer_does_not_break_the_stage(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повтор пары `(type, direction)` отбрасывается, а не падает вставкой.

    §3.1 держит частичный уникальный индекс по слоту, и второй вопрос в том
    же слоте уронил бы `IntegrityError`, потеряв заодно первый — транзакция
    стадии откатилась бы целиком, и факт остался бы без вопросов вовсе.
    """
    material_id = await _material(engine)
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections", "facts")
    )

    same_slot = {
        "type": "mcq",
        "direction": "forward",
        "difficulty": 2,
        "payload": {"question": "Что верно?", "options": ["а", "б"]},
        "answer": {"correct_index": 0},
    }
    provider = ScriptedProvider(answers={"questions": {"questions": [same_slot, same_slot]}})
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("questions",), provider=provider
    )

    rows = await _rows(
        engine,
        select(Question.fact_id, func.count(Question.id))
        .join(Fact, Fact.id == Question.fact_id)
        .where(Fact.material_id == material_id)
        .group_by(Question.fact_id),
    )
    assert rows
    assert all(int(count) == 1 for _, count in rows)


async def test_repeated_questions_stage_does_not_duplicate(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повторный прогон стадии не удваивает вопросы.

    §4.2: стадия удаляет свои прошлые артефакты до записи новых. Без этого
    второй прогон упал бы на уникальном индексе слота.
    """
    material_id = await _material(engine)
    stages = ("extract", "sections", "facts", "questions")
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, stages)
    first = await _rows(engine, select(func.count()).select_from(Question))

    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("questions",))
    second = await _rows(engine, select(func.count()).select_from(Question))

    assert first == second


# --- notes ----------------------------------------------------------------


async def test_note_is_rendered_with_the_structure_of_section_8(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Шаблон §8: заголовок темы, `summary_md`, четыре блока из фактов."""
    material_id = await _material(engine)
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections", "facts")
    )

    async with session_scope(sessions) as session:
        document = await render_note(session, material_id, _settings())

    assert document.startswith("## ")
    assert "### Ключевые понятия" in document
    assert "### Основные положения" in document
    assert "### Что запомнить" in document


async def test_note_marks_facts_that_will_never_be_asked(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """`⚠️` у выведенных и слабых фактов — §8.

    Пометка берётся из того же предиката §7.3, что отбор на генерацию. Иначе
    пользователь видел бы факт без пометки, по которому его никогда не
    спросят, и считал бы это сбоем планировщика.
    """
    material_id = await _material(engine)
    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections"))
    real = [int(row[0]) for row in await _rows(engine, select(Fragment.id).order_by(Fragment.ord))]

    provider = ScriptedProvider(
        answers={
            "facts": {
                "title": "Тема",
                "summary_md": "Описание.",
                "facts": [
                    {
                        "statement": "Подтверждённый факт",
                        "detail": "Раскрытие.",
                        "kind": "definition",
                        "fragment_ids": [real[0]],
                        "importance": 3,
                        "difficulty": 2,
                        "derived": False,
                        "confidence": 0.9,
                    },
                    {
                        "statement": "Вывод модели",
                        "detail": "Раскрытие.",
                        "kind": "definition",
                        "fragment_ids": [real[0]],
                        "importance": 3,
                        "difficulty": 2,
                        "derived": True,
                        "confidence": 0.9,
                    },
                ],
            }
        }
    )
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("facts",), provider=provider
    )

    async with session_scope(sessions) as session:
        document = await render_note(session, material_id, _settings())

    marked = [line for line in document.splitlines() if WARNING_MARK in line]
    assert any("Вывод модели" in line for line in marked)
    assert not any("Подтверждённый факт" in line for line in marked)


async def test_notes_stage_stores_nothing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Стадия конспекта не меняет базу: конспект — функция от фактов (§8).

    Любая его копия в базе разошлась бы с фактами после первой перегенерации.
    Проверяется, что стадия не затирает `summary_md`, полученный от §7.1, —
    именно это делала заглушка WP-03, и с настоящими фактами она уничтожала
    бы содержательное описание секции.
    """
    material_id = await _material(engine)
    await _run_stages(
        sessions, MaterialStorage(tmp_path), material_id, ("extract", "sections", "facts")
    )
    before = await _rows(
        engine,
        select(Section.summary_md, func.count(Fact.id))
        .join(Fact, Fact.section_id == Section.id)
        .where(Section.material_id == material_id)
        .group_by(Section.summary_md),
    )

    await _run_stages(sessions, MaterialStorage(tmp_path), material_id, ("notes",))

    after = await _rows(
        engine,
        select(Section.summary_md, func.count(Fact.id))
        .join(Fact, Fact.section_id == Section.id)
        .where(Section.material_id == material_id)
        .group_by(Section.summary_md),
    )
    assert before == after


# --- §6.4 на настоящих стадиях --------------------------------------------


async def test_every_stage_call_lands_in_llm_calls(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Критерий приёмки пакета: все вызовы попадают в `llm_calls` (§6.4).

    Проверяется на настоящих стадиях и настоящем `DatabaseRecorder`, а не на
    тестовых вызовах: именно поэтому пакет идёт одним PR, а не двумя.
    Назначения сверяются по составу — строка на стадию, и ни одной лишней.
    """
    material_id = await _material(engine)
    provider = ScriptedProvider(name="manual", model="scripted-1")
    recorder = DatabaseRecorder(sessions)
    await _run_stages(
        sessions,
        MaterialStorage(tmp_path),
        material_id,
        ("extract", "sections", "facts", "notes", "questions"),
        provider=provider,
        recorder=recorder,
    )

    rows = await _rows(
        engine,
        select(LlmCall.purpose, LlmCall.prompt_version, LlmCall.provider, LlmCall.ok),
    )
    purposes = [str(purpose) for purpose, _, _, _ in rows]
    assert len(rows) == len(provider.prompts)
    assert purposes.count("sections") == 1
    assert purposes.count("facts") == 1
    assert purposes.count("notes") == 0, "стадия конспекта модель не вызывает (§8)"
    assert purposes.count("questions") == 3, "по вызову на факт (§10.1)"
    assert {str(version) for _, version, _, _ in rows} == {
        "sections_v1",
        "facts_v1",
        "questions_v1",
    }
    assert all(bool(ok) for _, _, _, ok in rows)


async def test_llm_calls_rows_carry_the_material_and_user(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§3: `llm_calls.user_id` и `material_id`.

    Без них §6.4 показывает общий расход и не отвечает на вопрос «сколько
    стоил этот материал» — то есть не даёт сравнить факт с оценкой, которая
    хранится у материала.
    """
    material_id = await _material(engine)
    await _run_stages(
        sessions,
        MaterialStorage(tmp_path),
        material_id,
        ("extract", "sections"),
        recorder=DatabaseRecorder(sessions),
    )

    rows = await _rows(engine, select(LlmCall.material_id, LlmCall.user_id))
    assert rows
    assert all(int(material) == material_id for material, _ in rows)
    assert all(user is not None for _, user in rows)


async def test_accounting_row_survives_a_failing_stage(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Стадия упала **после** вызова — строка учёта осталась. §6.4.

    Это свойство, ради которого `DatabaseRecorder` открывает свою сессию, и
    до ревью PR #27 оно не было проверено ни одним тестом: оба теста про
    `llm_calls` шли по успешному пути, где откатывать нечего. Мутация
    «recorder пишет в сессию стадии» прошла бы незамеченной.

    Сценарий достижим и наблюдался на живом прогоне `make seed`: ответ для
    первой секции был готов, для второй нет, стадия `facts` упала, факты
    первой секции откатились — а строка о сделанном вызове осталась. Иначе
    §6.4 терял бы расход на вызовы, результат которых не сохранился, то есть
    ровно те, за которые заплачено зря.

    Проверяется в обе стороны: своя запись стадии исчезла, строка учёта
    осталась. Без первой половины тест прошёл бы и при транзакции, которая
    вообще не откатывается.
    """
    material_id = await _material(engine)
    client = fake_client(
        _settings(),
        provider=ScriptedProvider(name="manual"),
        recorder=DatabaseRecorder(sessions),
    )

    with pytest.raises(RuntimeError, match="стадия упала"):
        async with session_scope(sessions) as session:
            await client.structured(
                "[1] текст фрагмента",
                SectionsAnswer,
                purpose="sections",
                prompt_version="sections_v1",
                material_id=material_id,
            )
            session.add(
                Section(
                    material_id=material_id,
                    ord=1,
                    title="Секция, которой не будет",
                    summary_md="",
                    fragment_ids=[1],
                )
            )
            await session.flush()
            raise RuntimeError("стадия упала после вызова модели")

    sections = await _rows(engine, select(Section.title).where(Section.material_id == material_id))
    calls = await _rows(
        engine, select(LlmCall.purpose, LlmCall.ok).where(LlmCall.material_id == material_id)
    )
    assert sections == [], "запись стадии не откатилась — тест проверяет не то, что должен"
    assert [(str(purpose), bool(ok)) for purpose, ok in calls] == [("sections", True)]
