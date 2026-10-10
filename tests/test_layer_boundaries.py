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
