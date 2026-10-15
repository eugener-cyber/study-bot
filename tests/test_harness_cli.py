"""Консольный харнесс: `session` и `answer` как команды. План §1.4.

Тесты идут **через `main`** с подменённым `argv`, а не через функции напрямую:
«почти все проверки „Готово, если“ выполняются через харнесс», то есть
проверяется ровно то, что увидит человек, — разбор `argparse`, печать задания
и код возврата. До WP-05 у `core/harness.py` не было ни одного теста.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from core import harness
from core.config import Settings
from core.db.engine import build_engine
from core.db.models import Base, Question
from tests.conftest import ENV_STUB, _test_url
from tests.learning_factory import Scaffold, add_fact, add_question, add_review_state, scaffold

pytestmark = pytest.mark.db

TG_ID = 111
"""Владелец в тестах. Тот же, что первым идёт в `ALLOWED_USER_IDS` стенда."""


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Нацеливает харнесс на тестовую базу.

    Подменяется `load_settings`, а не окружение: харнесс читает настройки
    один раз на команду, и подмена на этом уровне оставляет настоящим всё
    остальное — разбор аргументов, печать, коды возврата.
    """
    import core.harness as harness

    settings = Settings(  # type: ignore[arg-type]
        _env_file=None,
        **{**ENV_STUB, "DATABASE_URL": _test_url(), "ALLOWED_USER_IDS": str(TG_ID)},
    )
    monkeypatch.setattr(harness, "load_settings", lambda: settings)


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Запускает `main` с заданной командной строкой, возвращает код выхода."""
    import core.harness as harness

    monkeypatch.setattr("sys.argv", ["harness", *argv])
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    return int(exit_info.value.code or 0)


async def _seed(count: int) -> Scaffold:
    """Чистая база и `count` обучаемых фактов, которым пора.

    Своим движком, а не фикстурой `engine`: тесты здесь **синхронные** —
    `main` сам вызывает `asyncio.run`, и из асинхронного теста это
    невозможно. Поэтому подготовка данных идёт отдельным `asyncio.run`.
    """
    engine = build_engine(_test_url())
    try:
        names = ", ".join(f'"{name}"' for name in Base.metadata.tables)
        async with engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        base = await scaffold(engine, tg_id=TG_ID)
        # Срок — относительно **настоящего** времени, а не опорного `NOW`
        # фабрик: харнесс берёт `datetime.now(UTC)`, и факт со сроком в
        # будущем для него не наступил (§16, §0.2 — досрочных повторений нет).
        due_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
        for index in range(count):
            fact_id = await add_fact(engine, base, statement=f"Факт {index}")
            await add_question(engine, fact_id)
            await add_review_state(engine, base.user_id, fact_id, due_at=due_at)
        return base
    finally:
        await engine.dispose()


async def _total_shown() -> int:
    engine = build_engine(_test_url())
    try:
        async with engine.connect() as connection:
            return int(
                (
                    await connection.execute(
                        select(func.coalesce(func.sum(Question.shown_count), 0))
                    )
                ).scalar_one()
            )
    finally:
        await engine.dispose()


@pytest.fixture
def prepared(_database: None, cli: None) -> Scaffold:
    """Три обучаемых факта, которым пора. Синхронная фикстура."""
    return asyncio.run(_seed(3))


@pytest.fixture
def prepared_single(_database: None, cli: None) -> Scaffold:
    """Один факт — меньше `MIN_SESSION_ITEMS`."""
    return asyncio.run(_seed(1))


def test_session_prints_the_task_with_numbered_options(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run(monkeypatch, "session")
    printed = capsys.readouterr().out

    assert code == 0
    assert "задание 1 из 3" in printed
    assert "Какой инструмент размечает ткань?" in printed
    assert "  1. мел" in printed
    assert "make answer OPTION=1..3" in printed


def test_answer_prints_the_explanation(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Критерий приёмки WP-05: `make answer` печатает разбор §22."""
    _run(monkeypatch, "session")
    capsys.readouterr()

    code = _run(monkeypatch, "answer", "--option", "1")
    printed = capsys.readouterr().out

    assert code == 0
    assert "✅ Верно" in printed
    assert "Правильный ответ: мел" in printed
    assert "Почему: Меловая линия видна" in printed
    assert "В источнике: «Мел размечает ткань" in printed
    assert "задание 2 из 3" in printed, "следующее задание не выдано"


def test_last_answer_prints_the_summary(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§22.1: счёт, время и список ошибок в конце сеанса."""
    _run(monkeypatch, "session")
    _run(monkeypatch, "answer", "--option", "1")
    _run(monkeypatch, "answer", "--idk")
    capsys.readouterr()

    code = _run(monkeypatch, "answer", "--option", "1")
    printed = capsys.readouterr().out

    assert code == 0
    assert "— Сеанс завершён —" in printed
    assert "Счёт: 2 из 3" in printed
    assert "Ошибки:" in printed


def test_answer_without_a_session_says_so(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Отказ внятный и с кодом 1: цель вызывается из скриптов."""
    code = _run(monkeypatch, "answer", "--option", "1")
    printed = capsys.readouterr().out

    assert code == 1
    assert "Активного сеанса нет" in printed


def test_session_without_candidates_says_so(
    prepared_single: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§16: кандидатов меньше порога — сеанс не создаётся, и это не ошибка."""
    code = _run(monkeypatch, "session")
    printed = capsys.readouterr().out

    assert code == 1
    assert "Повторять нечего" in printed
    assert "MIN_SESSION_ITEMS = 3" in printed


def test_both_option_and_idk_are_refused(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """«Не знаю» — отдельный исход §13.2, а не вариант ответа.

    Если бы команда принимала оба, в журнале `verdict` перестал бы отличать
    пропуск от ошибки — а на этой разнице стоят метрики §25.
    """
    _run(monkeypatch, "session")

    with pytest.raises(SystemExit) as exit_info:
        monkeypatch.setattr("sys.argv", ["harness", "answer", "--option", "1", "--idk"])
        import core.harness as harness

        harness.main()

    assert "OPTION" in str(exit_info.value)


def test_repeated_session_does_not_count_a_second_show(
    prepared: Scaffold,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Повторный `make session` печатает то же задание и не растит `shown_count`.

    §13.3 сортирует вопросы по числу показов, и перезагрузка экрана не должна
    влиять на порядок выдачи.
    """
    _run(monkeypatch, "session")
    first = capsys.readouterr().out
    _run(monkeypatch, "session")
    second = capsys.readouterr().out

    assert "задание 1 из 3" in first
    assert "задание 1 из 3" in second

    assert asyncio.run(_total_shown()) == 1, "повторный показ вырос в счётчике"


async def _material_with_pdf_fragments(pending: list[int] | None = None) -> int:
    """Материал типа `pdf` с фрагментами и, если нужно, пропущенными страницами."""
    engine = build_engine(_test_url())
    try:
        names = ", ".join(f'"{name}"' for name in Base.metadata.tables)
        async with engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        base = await scaffold(engine, tg_id=TG_ID)
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE materials SET kind='pdf', origin=CAST(:origin AS jsonb) WHERE id=:id"),
                {
                    "origin": json.dumps(
                        {"sha256": "abc", "pages": 3, "vision_pending_pages": pending or []}
                    ),
                    "id": base.material_id,
                },
            )
            await connection.execute(
                text(
                    "INSERT INTO fragments (material_id, ord, kind, locator, text, quality, tokens)"
                    " VALUES (:m, 2, 'text', CAST(:loc AS jsonb), :t, 1.0, 42)"
                ),
                {
                    "m": base.material_id,
                    "loc": json.dumps(
                        {
                            "type": "pdf",
                            "page": 2,
                            "bbox": [72, 100, 523, 200],
                            "heading": "Раскрой",
                        }
                    ),
                    "t": "Мел размечает ткань по лицевой стороне.",
                },
            )
        return base.material_id
    finally:
        await engine.dispose()


async def _material_with_asset(
    *, path_in_db: str, measurement: dict[str, object] | None = None
) -> int:
    """Материал с одной строкой `media_assets` и заданным путём в ней.

    `measurement` — запись перечня `origin.assets.items` для этого ассета:
    страница, доля площади, подпись, группа. Без неё перечня нет вовсе, и
    `dump-material` обязан это показать, а не промолчать.
    """
    engine = build_engine(_test_url())
    try:
        names = ", ".join(f'"{name}"' for name in Base.metadata.tables)
        async with engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        base = await scaffold(engine, tg_id=TG_ID)
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE materials SET kind='pdf' WHERE id=:id"),
                {"id": base.material_id},
            )
            asset_id = (
                await connection.execute(
                    text(
                        "INSERT INTO media_assets (material_id, path, mime, w, h, caption)"
                        " VALUES (:m, :p, 'image/png', 10, 20, :c) RETURNING id"
                    ),
                    {"m": base.material_id, "p": path_in_db, "c": "Рис. 1. Подпись"},
                )
            ).scalar_one()
            if measurement is not None:
                await connection.execute(
                    text(
                        # COALESCE обязателен: у `scaffold` origin пуст, а в Postgres
                        # `NULL || jsonb` даёт NULL — измерение не записалось бы молча.
                        "UPDATE materials SET origin ="
                        " COALESCE(origin, '{}'::jsonb) || CAST(:extra AS jsonb)"
                        " WHERE id=:id"
                    ),
                    {
                        "extra": json.dumps(
                            {"assets": {"items": [{"asset_id": asset_id, **measurement}]}}
                        ),
                        "id": base.material_id,
                    },
                )
        return int(base.material_id)
    finally:
        await engine.dispose()


def test_dump_material_shows_page_heading_and_quality(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Критерий WP-06: `make dump-material` показывает результат читаемо.

    Проверяется через `main` с подменённым `argv` — то есть ровно то, что
    увидит человек, включая код возврата.
    """
    material_id = asyncio.run(_material_with_pdf_fragments())

    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert f"материал {material_id}: тип=pdf" in printed
    assert "стр. 2" in printed
    assert "[Раскрой]" in printed
    assert "q=1.0" in printed
    assert "Мел размечает ткань" in printed


def test_dump_material_names_pages_awaiting_vision(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§6.3: последствия обязаны быть видимыми, а не выводимыми.

    До PR B пакета страницы-сканы пропускаются. Молчаливый вывод означал бы
    материал неизвестного состава: человек видит фрагменты и не знает, что
    третья страница в них не попала.
    """
    material_id = asyncio.run(_material_with_pdf_fragments(pending=[1, 3]))

    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert "страниц без текстового слоя: 2" in printed
    assert "1, 3" in printed


def test_dump_material_prints_page_area_and_group(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Критерий пакета: набор виден в выводе, а не только число ассетов.

    Измерения (страница, доля площади, группа) лежат в `materials.origin` —
    колонок под них в `media_assets` нет. Первая редакция `_assets_block` их
    **не читала**, хотя докстрока утверждала обратное, и приёмка «группа из
    десяти на стр. 10 видна в `dump-material`» не выполнялась: тесты
    проверяли только число ассетов и пометку о файле. Нашло ревью PR #53.
    """
    material_id = asyncio.run(
        _material_with_asset(
            path_in_db="7/assets/img001.png",
            measurement={
                "page": 10,
                "area_ratio": 0.0295,
                "caption": "Рис. 1. Виды",
                "group_id": "page10-fig1",
            },
        )
    )

    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert "стр. 10" in printed, "страница не напечатана"
    assert "0.0295" in printed, "доля площади не напечатана"
    assert "[page10-fig1]" in printed, "группа не напечатана"


def test_dump_material_shows_an_asset_missing_from_origin(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Расхождение между таблицей и `origin` видно, а не проглочено.

    Строка `media_assets` без записи в перечне — это рассогласование, и ему
    место в выводе: «стр. —». Отрицательная половина к тесту выше.
    """
    material_id = asyncio.run(_material_with_asset(path_in_db="7/assets/img001.png"))

    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert "стр. —" in printed


def test_dump_material_marks_an_asset_whose_file_is_absent(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Пометка «файла нет по пути» — обнаружение окна §4.2.

    Строки `media_assets` коммитятся до переноса файлов, и падение процесса
    между этими шагами оставляет ссылку без файла. Порядок общий для всех
    стадий с артефактами, менять его ради одного случая нельзя — поэтому
    последствие показывается.

    Прогон идёт **из другого рабочего каталога** (`monkeypatch.chdir`): путь в
    базе относителен корня хранилища, и в своём каталоге он совпал бы с
    рабочим случайно, а проверка ничего бы не значила.
    """
    material_id = asyncio.run(_material_with_asset(path_in_db="999/assets/нет.png"))

    monkeypatch.chdir(tmp_path)
    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert "ассетов: 1" in printed
    assert "файла нет по пути" in printed


def test_dump_material_does_not_mark_an_asset_that_exists(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Обратное направление: у существующего файла пометки нет.

    Иначе предыдущий тест прошёл бы и у реализации, которая печатает пометку
    всегда. Файл создаётся по тому же относительному пути от корня
    хранилища, что лежит в базе, — то есть проверяется склейка, а не строка.
    """
    relative = "7/assets/img001.png"
    material_id = asyncio.run(_material_with_asset(path_in_db=relative))

    root = tmp_path / "materials"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "MATERIALS_ROOT", str(root))
    code = _run(monkeypatch, "dump-material", str(material_id))
    printed = capsys.readouterr().out

    assert code == 0
    assert "ассетов: 1" in printed
    assert "файла нет по пути" not in printed


def test_dump_material_of_a_missing_material_says_so(
    _database: None,
    cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _run(monkeypatch, "dump-material", "999999")

    assert code == 1
    assert "нет" in capsys.readouterr().out
