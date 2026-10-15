"""Границы слоёв держатся проверкой, а не соглашением. ТЗ §35, §2.1.

Тот же приём, что grep на `create_all` и на `teachable_facts(` в WP-02:
договорённость, которую не проверяет прогон, живёт до первого спешного
коммита.

Проверяется по исходному тексту через `ast`, а не по загруженным модулям:
импорт внутри функции (отложенный) рантайму виден не всегда, а дереву разбора
виден всегда.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

LLM = ROOT / "core" / "llm"
LEARNING = ROOT / "core" / "learning"
CORE = ROOT / "core"
HANDLERS = ROOT / "core" / "ingest" / "handlers.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("module", sorted(LLM.glob("*.py")), ids=lambda path: path.name)
def test_llm_layer_does_not_import_the_ingest_package(module: Path) -> None:
    """Направление зависимости — `ingest → llm`, и только оно.

    Таблица коэффициентов и цены переехали в `core/llm/estimation.py` именно
    поэтому: их второй потребитель — клиент, и он тянул бы за собой весь
    пакет обработки материала. В WP-10 и WP-11 валидатор §11.4 и судья §21.3
    потянули бы `core/ingest` за собой, не имея к обработке материала
    отношения.
    """
    offending = {name for name in _imports(module) if name.startswith("core.ingest")}
    assert not offending, f"{module.name} импортирует {sorted(offending)}"


def test_handlers_see_only_the_client() -> None:
    """§2.1: «бизнес-логика этих данных не видит».

    Стадия не должна знать ни провайдера, ни режимов, ни базовых типов слоя:
    запись в `llm_calls` выполняет клиент. Импорт `core.llm.base` в стадии
    означал бы, что она ловит `SchemaError` сама — то есть дублирует правила
    ретраев §6.1 там, где они уже реализованы.
    """
    imported = _imports(HANDLERS)
    forbidden = {
        "core.llm.base",
        "core.llm.manual",
        "core.llm.factory",
        "core.llm.recording",
    }
    assert not (imported & forbidden), f"стадии импортируют {sorted(imported & forbidden)}"
    assert "core.llm.client" in imported


def test_handlers_do_not_inline_the_confidence_threshold() -> None:
    """Порог обучаемости приходит из настроек, а не литералом.

    Преамбула §30.3: «константы конфигурации, а не литералы в коде». Литерал
    здесь означал бы, что оператор правит `.env`, а генерация вопросов
    продолжает работать по старому числу — молча.
    """
    source = HANDLERS.read_text(encoding="utf-8")
    assert "MIN_FACT_CONFIDENCE" in source
    for literal in ("0.55", "0.65", "0.7"):
        assert literal not in source, f"порог {literal} вписан литералом"


def test_prompt_files_are_not_embedded_in_code() -> None:
    """Промпты живут файлами, как требует §6.1.

    Промпт в тройных кавычках внутри модуля ломает две вещи сразу: версия
    перестаёт быть частью имени файла (а значит, ключ записи не меняется при
    её смене), и дифф правки промпта читается как перемешанные строки кода и
    текста.
    """
    source = HANDLERS.read_text(encoding="utf-8")
    assert "load_prompt" in source
    assert "Ответ — JSON по схеме" not in source


@pytest.mark.parametrize("module", sorted(LEARNING.glob("*.py")), ids=lambda path: path.name)
def test_learning_layer_knows_nothing_about_telegram_or_ingest(module: Path) -> None:
    """Направление зависимости — `bot → core.services → core.learning`.

    Педагогический слой не должен знать ни способа, которым задание попало к
    человеку, ни конвейера обработки материала. Иначе консольный путь §1.4
    («почти все проверки „Готово, если“ выполняются через харнесс») перестал
    бы существовать: любая проверка планировщика тянула бы aiogram.

    `core.services` в запрете тоже: сервис вызывает слой, а не наоборот.
    Обратный импорт дал бы цикл — `services.sessions` импортирует
    `learning.planner` ради `PLANNER_VERSION` и `plan_items`.
    """
    forbidden = ("aiogram", "bot", "core.ingest", "core.services", "arq")
    offending = {
        name
        for name in _imports(module)
        if any(name == item or name.startswith(f"{item}.") for item in forbidden)
    }
    assert not offending, f"{module.name} импортирует {sorted(offending)}"


def test_planner_uses_the_teachable_wrapper_not_the_sql_function() -> None:
    """§38 №18 в той части, которая доказуема в WP-05.

    Свойство целиком («предикат один и тот же в трёх местах») доказуемо только
    в WP-16, когда появится знаменатель coverage. Здесь проверяется то, что
    доступно: планировщик — второй потребитель — берёт предикат обёрткой из
    `core/db/teachable.py`, а не пишет собственное условие.

    `test_threshold_not_inlined_by_consumers` в `tests/test_teachable.py`
    запрещает прямой вызов SQL-функции по всему `core/`; этот тест — про
    обратное: что обёртка действительно импортирована. Без него планировщик
    мог бы собрать своё условие на `facts` и пройти оба запрета.
    """
    imported = _imports(LEARNING / "planner.py")
    assert "core.db.teachable" in imported, "планировщик не берёт предикат §14.3 из обёртки"


def test_pymupdf_lives_in_one_module() -> None:
    """PyMuPDF разрешён ровно в `core/adapters/pdf.py`. ТЗ §35 (v3.15).

    §35 даёт послабление `disallow_untyped_calls = false` этому модулю
    **«для него одного»** и требует теста против расползания — вот он.
    Второй модуль с этим импортом либо потребует второго послабления, либо
    упрётся в `--strict`, и оба исхода обнаружатся здесь, а не в споре о том,
    что имелось в виду.

    Замер, по которому принято решение: пробный файл из 18 строк даёт 4 ошибки
    `--strict`; модуль рендера того же вида — ещё 3 (`pymupdf.Rect` и
    `Pixmap.tobytes`). Поэтому рендер и подсветка — методы `PdfAdapter`, а не
    отдельный модуль.
    """
    allowed = CORE / "adapters" / "pdf.py"
    offending: list[str] = []
    for module in sorted(CORE.rglob("*.py")):
        if module == allowed:
            continue
        if any(name == "pymupdf" or name.startswith("pymupdf.") for name in _imports(module)):
            offending.append(str(module.relative_to(ROOT)))

    assert not offending, f"PyMuPDF вышел за границу послабления §35: {offending}"


def test_mypy_relaxation_covers_exactly_one_module() -> None:
    """Послабление в `pyproject.toml` объявлено ровно для одного модуля.

    Проверяется сам конфиг, а не только импорты: расползание бывает двух
    видов — новый модуль с библиотекой и новая строка в списке `module`.
    Первое держит тест выше, второе — этот.
    """
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    relaxed = [
        override
        for override in config["tool"]["mypy"]["overrides"]
        if override.get("disallow_untyped_calls") is False
    ]

    assert len(relaxed) == 1, f"послаблений не одно: {relaxed}"
    assert relaxed[0]["module"] == ["core.adapters.pdf"], relaxed[0]["module"]


def test_assets_is_optional_in_the_contract() -> None:
    """§5.1 v3.17: третий метод необязательный — адаптер без медиафайлов цел.

    Проверяется на **минимальном** адаптере, объявленном здесь же: он
    реализует только `probe` и `extract`. Первая редакция теста смотрела в
    `StubAdapter.__dict__`, и когда заглушка получила `assets` ради тестов
    стадии, проверка сломалась — она держалась случайного факта о заглушке, а
    не на контракте. Если метод станет `@abstractmethod`, этот класс
    перестанет создаваться, и тест покраснеет здесь.
    """
    from collections.abc import AsyncIterator

    from core.adapters.base import ExtractedFragment, IngestCtx, SourceAdapter, SourceMeta

    class Minimal(SourceAdapter):
        kind = "minimal"

        async def probe(self, path: str) -> SourceMeta:
            return SourceMeta()

        async def extract(self, path: str, ctx: IngestCtx) -> AsyncIterator[ExtractedFragment]:
            empty: tuple[ExtractedFragment, ...] = ()
            for fragment in empty:
                yield fragment

    adapter = Minimal()

    assert "assets" not in Minimal.__dict__, "минимальный адаптер переопределил метод"
    assert hasattr(adapter, "assets")


async def test_default_assets_yields_nothing_and_is_an_iterator() -> None:
    """Пустая реализация — асинхронный итератор, а не корутина.

    Разница существенна: `async def` без `yield` вернул бы корутину, и
    `async for` по ней упал бы в стадии, а не здесь.
    """
    from core.adapters.base import IngestCtx
    from core.adapters.stub import StubAdapter

    # Заглушка без заданных ассетов: её `assets` отдаёт пустой поток, как и
    # реализация по умолчанию. Сама реализация по умолчанию проверена в тесте
    # выше на минимальном адаптере.
    stream = StubAdapter().assets("/tmp/нет", IngestCtx(material_id=1, tmp_dir="/tmp"))

    assert hasattr(stream, "__anext__"), "метод вернул не асинхронный итератор"
    assert [item async for item in stream] == []
