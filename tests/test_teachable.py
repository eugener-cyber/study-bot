"""Предикат обучаемости §14.3 — флаги и арифметика.

Две группы, и это сознательно. Параметризация по пяти флагам даёт 64 клетки,
но из них одна истинная и пять содержательно ложных: остальные 58 создают
впечатление исчерпанности, не добавляя проверок. Ревью плана указало на это
прямо, поэтому рядом стоит вторая группа — там, где предикат ломается молча:
арифметика `final_confidence`, равенство порогу, `NULL` в `quality`, чужой
фрагмент, отсутствие evidence.

Два теста здесь важнее остальных. `test_threshold_comes_from_configuration`
проверяет, что порог действительно параметр функции, а не литерал в DDL.
`test_fact_without_evidence_is_never_teachable` закрывает дефект, найденный
мутационным прогоном уже после написания функции: при нулевом пороге факт без
evidence становился обучаемым, то есть нарушался И-1.
"""

from __future__ import annotations

import datetime
import itertools
import pathlib

import pytest
from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncEngine

from core.db.models import Fact, Fragment, Material, Question, Section, User
from core.db.teachable import FUNCTION_NAME, teachable_facts_select

pytestmark = pytest.mark.db

GENERATION_STATUSES = ("pending", "ok", "single_format", "unavailable")
"""§3.3. `unavailable` — единственное значение, исключающее факт."""

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _scaffold(engine: AsyncEngine) -> tuple[int, int]:
    """Пользователь, материал и секция — минимум, на котором держится факт."""
    async with engine.begin() as connection:
        user_id = (
            await connection.execute(
                insert(User).values(tg_id=1, created_at=NOW).returning(User.id)
            )
        ).scalar_one()
        material_id = (
            await connection.execute(
                insert(Material)
                .values(user_id=user_id, kind="pdf", status="ready", created_at=NOW)
                .returning(Material.id)
            )
        ).scalar_one()
        section_id = (
            await connection.execute(
                insert(Section)
                .values(material_id=material_id, ord=1, title="т")
                .returning(Section.id)
            )
        ).scalar_one()
    return material_id, section_id


async def _fragment(engine: AsyncEngine, material_id: int, quality: float | None) -> int:
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(Fragment)
                    .values(
                        material_id=material_id,
                        ord=1,
                        kind="text",
                        locator={"type": "pdf", "page": 1},
                        quality=quality,
                    )
                    .returning(Fragment.id)
                )
            ).scalar_one()
        )


async def _fact(
    engine: AsyncEngine,
    section_id: int,
    material_id: int,
    *,
    fragment_ids: list[int],
    confidence: float,
    suspended: bool = False,
    derived: bool = False,
    generation_status: str = "ok",
) -> int:
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(Fact)
                    .values(
                        material_id=material_id,
                        section_id=section_id,
                        statement="у",
                        detail="д",
                        kind="definition",
                        fragment_ids=fragment_ids,
                        confidence=confidence,
                        suspended=suspended,
                        derived=derived,
                        generation_status=generation_status,
                        created_at=NOW,
                    )
                    .returning(Fact.id)
                )
            ).scalar_one()
        )


async def _question(engine: AsyncEngine, fact_id: int, status: str) -> None:
    async with engine.begin() as connection:
        await connection.execute(
            insert(Question).values(
                fact_id=fact_id,
                type="mcq",
                direction="forward",
                payload={},
                answer={},
                difficulty=2,
                status=status,
                created_at=NOW,
            )
        )


async def _teachable_ids(engine: AsyncEngine, threshold: float) -> set[int]:
    async with engine.connect() as connection:
        result = await connection.execute(teachable_facts_select(threshold))
        return {int(row.fact_id) for row in result}


# --- Группа 1: комбинации флагов §14.3 -------------------------------------

CASES = list(
    itertools.product(
        (False, True),  # suspended
        (False, True),  # derived
        (True, False),  # confidence выше порога
        GENERATION_STATUSES,
        (True, False),  # есть валидный вопрос
    )
)


@pytest.mark.parametrize(
    "suspended,derived,confident,generation_status,has_valid",
    CASES,
    ids=[f"s{int(s)}-d{int(d)}-c{int(c)}-{g}-q{int(q)}" for s, d, c, g, q in CASES],
)
async def test_flag_combinations(
    engine: AsyncEngine,
    clean_tables: None,
    suspended: bool,
    derived: bool,
    confident: bool,
    generation_status: str,
    has_valid: bool,
) -> None:
    """Все 64 комбинации пяти условий §14.3.

    Факт обучаем тогда и только тогда, когда выполнены все пять. Ожидание
    считается здесь из тех же пяти значений, а не перечисляется таблицей:
    таблица на 64 строки была бы вторым экземпляром предиката, и ошибка в ней
    совпала бы с ошибкой в SQL ровно в тех случаях, где я думал одинаково
    неверно дважды.
    """
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[fragment_id],
        confidence=0.9 if confident else 0.2,
        suspended=suspended,
        derived=derived,
        generation_status=generation_status,
    )
    await _question(engine, fact_id, "valid" if has_valid else "draft")

    expected = (
        not suspended
        and not derived
        and confident
        and generation_status != "unavailable"
        and has_valid
    )
    assert (fact_id in await _teachable_ids(engine, 0.55)) is expected


# --- Группа 2: арифметика final_confidence ---------------------------------


async def test_quality_is_minimum_not_average(engine: AsyncEngine, clean_tables: None) -> None:
    """`final_confidence` считается по МИНИМУМУ quality, а не по среднему.

    §7.3: `final_confidence = confidence × min(fragment.quality)`. При
    `confidence = 0.9` и фрагментах 1.0 и 0.3 минимум даёт 0.27 — ниже порога
    0.55, среднее дало бы 0.585 — выше. Подмена `MIN` на `AVG` прошла бы
    любую проверку из группы 1.
    """
    material_id, section_id = await _scaffold(engine)
    good = await _fragment(engine, material_id, 1.0)
    bad = await _fragment(engine, material_id, 0.3)
    fact_id = await _fact(engine, section_id, material_id, fragment_ids=[good, bad], confidence=0.9)
    await _question(engine, fact_id, "valid")

    assert fact_id not in await _teachable_ids(engine, 0.55)


@pytest.mark.parametrize("threshold", [0.55, 0.65, 0.7, 0.9])
async def test_fact_exactly_at_threshold_is_teachable(
    engine: AsyncEngine, clean_tables: None, threshold: float
) -> None:
    """§14.3 пишет `>=`: факт ровно на пороге обучаем.

    Прогоняется по четырём значениям, а не по дефолтному 0.55. С параметром
    `double precision` вместо `real` сравнение даёт ложь на 0.65, 0.7 и 0.9 —
    из-за ошибки представления float4, вскрываемой повышением типа. На 0.55
    ложь не возникает, поэтому тест на одном дефолте был бы зелёным по
    совпадению и покраснел бы у заказчика после правки порога в `.env`.

    Допуск вида `>= threshold - 1e-6` здесь запрещён: это скрытый шестой
    порог, которого в §30.3 нет.
    """
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=threshold
    )
    await _question(engine, fact_id, "valid")

    assert fact_id in await _teachable_ids(
        engine, threshold
    ), f"факт с final_confidence = {threshold} не обучаем при пороге {threshold}"


async def test_null_quality_does_not_make_fact_teachable(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """`quality` объявлен nullable с `DEFAULT 1.0` — `NULL` достижим.

    `MIN` по множеству с `NULL` его игнорирует, но множество из одного `NULL`
    даёт `NULL`, и без `COALESCE` всё произведение стало бы `NULL`. Факт тогда
    выпал бы из выборки — правильный исход, но достигнутый случайно.
    """
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, None)
    fact_id = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.9
    )
    await _question(engine, fact_id, "valid")

    assert fact_id not in await _teachable_ids(engine, 0.55)


@pytest.mark.parametrize("threshold", [0.55, 0.0])
async def test_fact_without_evidence_is_never_teachable(
    engine: AsyncEngine, clean_tables: None, threshold: float
) -> None:
    """Факт без evidence не обучаем **при любом пороге**, включая нулевой.

    Это найдено мутационным прогоном и исправлено. В первой версии функции
    `COALESCE(..., 0)` стоял в условии отбора, с обоснованием «факт без
    evidence отсекает порог». Обоснование верно только для положительного
    порога: при `MIN_FACT_CONFIDENCE = 0` произведение давало `0 >= 0`, и
    факт без evidence становился обучаемым — нарушение И-1 (§7.3 проверка 1).

    Ноль достижим: §30.3 прямо называет этот порог одним из тех, которые
    придётся крутить после первых материалов. Параметризация по двум
    значениям и нужна потому, что на 0.55 дефект не проявлялся.
    """
    material_id, section_id = await _scaffold(engine)
    fact_id = await _fact(engine, section_id, material_id, fragment_ids=[], confidence=1.0)
    await _question(engine, fact_id, "valid")

    assert fact_id not in await _teachable_ids(engine, threshold)


@pytest.mark.parametrize("threshold", [0.55, 0.0])
async def test_fact_with_missing_fragment_is_not_teachable(
    engine: AsyncEngine, clean_tables: None, threshold: float
) -> None:
    """Ссылка на несуществующий фрагмент — не evidence.

    Тот же класс, что пустой массив: `MIN` по несуществующим идентификаторам
    даёт `NULL`. Проверяется отдельно, потому что пустой массив и битая
    ссылка — разные пути к одному `NULL`, и защита от первого не обязана
    закрывать второй.
    """
    material_id, section_id = await _scaffold(engine)
    fact_id = await _fact(engine, section_id, material_id, fragment_ids=[999_999], confidence=1.0)
    await _question(engine, fact_id, "valid")

    assert fact_id not in await _teachable_ids(engine, threshold)


@pytest.mark.parametrize("threshold", [0.55, 0.0])
async def test_mixed_quality_ignores_the_unmeasured_fragment(
    engine: AsyncEngine, clean_tables: None, threshold: float
) -> None:
    """Фрагмент с `quality = NULL` рядом с измеренным **игнорируется**.

    Фиксирую фактическое поведение, а не желаемое. `MIN` в SQL пропускает
    `NULL`, поэтому факт с фрагментами 1.0 и `NULL` обучаем с
    `final_confidence = 1.0` — как будто второго фрагмента нет.

    Вырожденный случай (все фрагменты `NULL`) покрыт отдельным тестом и даёт
    исключение факта. Смесь не покрывал никто, хотя поведение было известно:
    докстрока того теста сама пишет, что `MIN` игнорирует `NULL`. Нашло ревью
    PR #10.

    **Правильно ли это — вопрос к Архитектору, а не правка здесь.** §7.3
    определяет `final_confidence = confidence × min(fragment.quality)` и не
    говорит, что делать с неизвестным качеством части evidence. Тест
    закрепляет текущее поведение, чтобы смена была сознательной, а не
    случайной; вопрос оформлен change request'ом.
    """
    material_id, section_id = await _scaffold(engine)
    measured = await _fragment(engine, material_id, 1.0)
    unmeasured = await _fragment(engine, material_id, None)
    fact_id = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[measured, unmeasured],
        confidence=1.0,
    )
    await _question(engine, fact_id, "valid")

    assert fact_id in await _teachable_ids(
        engine, threshold
    ), "поведение смеси изменилось — это сознательная правка или дефект?"


@pytest.mark.parametrize("threshold", [0.55, 0.0])
async def test_mixed_evidence_ignores_the_broken_reference(
    engine: AsyncEngine, clean_tables: None, threshold: float
) -> None:
    """Битая ссылка рядом с существующим фрагментом **игнорируется**.

    Тот же механизм, другой путь: `MIN` по множеству, где часть
    идентификаторов не находится, считается по найденным. Факт обучаем, хотя
    часть его evidence не существует.

    Вырожденный случай (все ссылки битые) исключает факт — это проверено
    отдельно. Смесь не проверял никто. Отдельный тест, потому что пустой
    массив и битая ссылка — разные пути к одному `NULL`, и защита от первого
    не обязана закрывать второй.

    Как и выше, поведение зафиксировано, а не исправлено: §7.3 проверка 1
    требует, чтобы `fact.fragment_ids ⊆ submitted_fragment_ids` проверялось на
    стадии извлечения, а не в предикате. Дублировать проверку здесь значило бы
    реализовать больше, чем требует §14.3.
    """
    material_id, section_id = await _scaffold(engine)
    existing = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[existing, 999_999],
        confidence=1.0,
    )
    await _question(engine, fact_id, "valid")

    assert fact_id in await _teachable_ids(
        engine, threshold
    ), "поведение смеси изменилось — это сознательная правка или дефект?"


async def test_returned_confidence_is_never_null(engine: AsyncEngine, clean_tables: None) -> None:
    """`final_confidence` приходит потребителям числом, а не `NULL`.

    Для этого COALESCE оставлен в списке колонок, хотя из условия отбора он
    убран. Три потребителя (§14.1, §16, §4.6) читают эту колонку, и `NULL`
    в арифметике coverage дал бы не ошибку, а тихо неверное значение.
    """
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.9
    )
    await _question(engine, fact_id, "valid")

    async with engine.connect() as connection:
        rows = (await connection.execute(teachable_facts_select(0.55))).all()
    assert [row.fact_id for row in rows] == [fact_id]
    assert rows[0].final_confidence is not None
    assert rows[0].final_confidence == pytest.approx(0.9, abs=1e-6)


async def test_fragment_of_another_material_is_counted_as_evidence(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """Соединение идёт по `fragment_ids`, а не по материалу.

    Фиксирую фактическое поведение: функция берёт фрагменты по
    идентификаторам, не проверяя принадлежность материалу. Проверку
    принадлежности выполняет §7.3 на стадии извлечения — там она и описана, и
    дублировать её в предикате значило бы реализовать больше, чем требует
    §14.3. Если это окажется неверным, тест покажет, что поведение менялось
    сознательно, а не случайно.
    """
    material_id, section_id = await _scaffold(engine)
    async with engine.begin() as connection:
        other_material = (
            await connection.execute(
                insert(Material)
                .values(user_id=1, kind="pdf", status="ready", created_at=NOW)
                .returning(Material.id)
            )
        ).scalar_one()
    alien = await _fragment(engine, int(other_material), 1.0)
    fact_id = await _fact(engine, section_id, material_id, fragment_ids=[alien], confidence=0.9)
    await _question(engine, fact_id, "valid")

    assert fact_id in await _teachable_ids(engine, 0.55)


async def test_threshold_comes_from_configuration(engine: AsyncEngine, clean_tables: None) -> None:
    """Смена порога меняет состав обучаемых фактов.

    Главный тест файла. Он доказывает, что порог — параметр функции, а не
    литерал в DDL миграции: иначе `MIN_FACT_CONFIDENCE` из `.env` менял бы
    поведение генерации вопросов (§7.3) и не менял бы ни coverage (§14.1), ни
    выборку планировщика (§16), и расхождение было бы невидимым.
    """
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.6
    )
    await _question(engine, fact_id, "valid")

    assert fact_id in await _teachable_ids(engine, 0.55), "при пороге 0.55 факт обучаем"
    assert fact_id not in await _teachable_ids(engine, 0.7), "при пороге 0.7 — уже нет"


async def test_rejected_question_does_not_make_fact_teachable(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """Нужен именно `valid`: `draft` и `rejected` не считаются (§3.3)."""
    material_id, section_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.9
    )
    await _question(engine, fact_id, "draft")
    await _question(engine, fact_id, "rejected")

    assert fact_id not in await _teachable_ids(engine, 0.55)


async def test_function_is_inlined_by_planner(engine: AsyncEngine, clean_tables: None) -> None:
    """`LANGUAGE sql STABLE` даёт встраивание, а не вызов на строку.

    Причина, по которой в плане выбрана set-returning функция, а не view:
    возражение «N вызовов» к этой форме не относится. Проверяется отсутствием
    узла `Function Scan` в плане запроса.
    """
    async with engine.connect() as connection:
        plan = (
            await connection.execute(
                text(f"EXPLAIN SELECT * FROM {FUNCTION_NAME}(CAST(0.55 AS real))")
            )
        ).all()
    rendered = "\n".join(str(row[0]) for row in plan)
    assert "Function Scan" not in rendered, rendered


def test_threshold_not_inlined_by_consumers() -> None:
    """Функция вызывается только из миграции и из обёртки.

    Замечание ревью плана: «точка композиции одна» было словесным правилом,
    таким же, как «только миграции». Ничто не мешало будущему потребителю
    вызвать `teachable_facts(0.55)` напрямую, и порог снова стал бы литералом,
    только уже в коде потребителя.
    """
    pattern = FUNCTION_NAME + "("
    allowed = {"core/db/teachable.py", "alembic/versions/0002_teachable.py"}
    root = pathlib.Path(__file__).resolve().parents[1]

    offenders: list[str] = []
    for folder in ("core", "bot", "alembic"):
        for path in root.glob(f"{folder}/**/*.py"):
            relative = str(path.relative_to(root))
            if relative in allowed:
                continue
            if pattern in path.read_text(encoding="utf-8"):
                offenders.append(relative)
    assert not offenders, f"{pattern} вызывается напрямую в: {offenders}"


# --- Группа 5: отбор на генерацию вопросов (§7.3 проверки 3 и 4) -----------


async def _generatable_ids(engine: AsyncEngine, threshold: float) -> set[int]:
    from core.db.teachable import generatable_facts_select

    async with engine.connect() as connection:
        result = await connection.execute(generatable_facts_select(threshold))
        return {int(row.fact_id) for row in result}


async def test_generatable_differs_from_teachable_exactly_by_valid_questions(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """Два предиката связаны, и связь держится проверкой, а не соглашением.

    `teachable_facts` требует валидного вопроса — для планировщика и
    статистики это правильно: факт без вопросов выдать нельзя. Стадия
    генерации работает **до** их появления, и обучаемость на этом шаге пуста
    по построению.

    Разница обязана быть ровно множеством фактов без валидных вопросов. Иначе
    при правке одного второй молча начнёт отбирать другое, и факты либо
    получат вопросы, но не попадут в расписание, либо наоборот — то есть
    сломается §4.6 или §16, а причина будет в соседнем предикате.
    """
    section_id, material_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)

    with_question = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.9
    )
    await _question(engine, with_question, "valid")
    without_question = await _fact(
        engine, section_id, material_id, fragment_ids=[fragment_id], confidence=0.9
    )

    teachable = await _teachable_ids(engine, 0.55)
    generatable = await _generatable_ids(engine, 0.55)

    assert teachable == {with_question}
    assert generatable == {with_question, without_question}
    assert generatable - teachable == {without_question}


@pytest.mark.parametrize(
    "derived,confidence,expected",
    [
        (False, 0.9, True),
        (True, 0.9, False),
        (False, 0.1, False),
        (False, 0.55, True),
    ],
    ids=["годный", "вывод-модели", "ниже-порога", "ровно-на-пороге"],
)
async def test_generatable_applies_checks_3_and_4_of_7_3(
    engine: AsyncEngine, clean_tables: None, derived: bool, confidence: float, expected: bool
) -> None:
    """§7.3: ниже порога — вопросы не генерируются; `derived` — не участвует в SRS.

    Случай «ровно на пороге» обязателен отдельной строкой: сравнение в
    `double precision` вместо `real` даёт на нём ложь, и это тот же дефект,
    который ловился в WP-02 на порогах 0.65, 0.7 и 0.9.
    """
    section_id, material_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[fragment_id],
        confidence=confidence,
        derived=derived,
    )

    assert (fact_id in await _generatable_ids(engine, 0.55)) is expected


async def test_generatable_ignores_generation_status(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """Факт со статусом `unavailable` остаётся кандидатом на перегенерацию.

    Запрет означал бы, что отказ окончателен, тогда как §12.2 обещает
    перегенерацию, а §38 №19 — возврат факта в ротацию автоматически.
    Обучаемость этот статус исключает, и правильно: без вопросов факт выдать
    нельзя. Два предиката расходятся здесь намеренно.
    """
    section_id, material_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)
    fact_id = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[fragment_id],
        confidence=0.9,
        generation_status="unavailable",
    )
    await _question(engine, fact_id, "valid")

    assert fact_id in await _generatable_ids(engine, 0.55)
    assert fact_id not in await _teachable_ids(engine, 0.55)


async def test_generatable_excludes_suspended_and_evidenceless_facts(
    engine: AsyncEngine, clean_tables: None
) -> None:
    """Приостановленный факт и факт без evidence вопросов не получают.

    Второй — защита И-1: генерировать вопрос по утверждению, которое ничем не
    подтверждено, значит создать задание, разбор которого невозможно
    сослать на источник.
    """
    section_id, material_id = await _scaffold(engine)
    fragment_id = await _fragment(engine, material_id, 1.0)

    suspended = await _fact(
        engine,
        section_id,
        material_id,
        fragment_ids=[fragment_id],
        confidence=0.9,
        suspended=True,
    )
    evidenceless = await _fact(engine, section_id, material_id, fragment_ids=[], confidence=0.9)

    generatable = await _generatable_ids(engine, 0.55)
    assert suspended not in generatable
    assert evidenceless not in generatable
