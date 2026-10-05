"""Модели сверяются с таблицей §3 ТЗ построчно.

Та же идея, что в `test_thresholds.py` из WP-01: спецификация читается как
источник, а не переписывается в тест руками. Переписанный руками список
колонок расходится с ТЗ так же молча, как расходились пороги, — и проверять
его было бы проверкой собственной памяти.

Разбор намеренно грубый: он не понимает SQL, а вытаскивает имя колонки,
пометки `NOT NULL`, `FK` и `NULL`. Больше и не нужно — остальное проверяет
контракт против живой базы в `test_schema_contract.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.db.models import Base

SPEC = Path(__file__).resolve().parents[1] / "docs" / "ТЗ.md"

# Колонки, которых в §3 нет под этими именами: §3 пишет `PRIMARY KEY (a, b)`
# отдельной строкой, а не пометкой у колонки.
_PK_LINE = re.compile(r"PRIMARY\s+KEY\s*\(", re.IGNORECASE)


def _spec_block() -> str:
    """SQL-блок раздела §3, без комментариев."""
    text = SPEC.read_text(encoding="utf-8")
    section = text.split("# 3. Модель данных")[1].split("## 3.1")[0]
    block = section.split("```sql")[1].split("```")[0]
    return re.sub(r"--[^\n]*", "", block)


def _split_top_level(body: str) -> list[str]:
    """Делит перечень колонок по запятым верхнего уровня."""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _spec_tables() -> dict[str, dict[str, str]]:
    """{таблица: {колонка: её объявление}} по §3."""
    block = _spec_block()
    tables: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"^(\w+)\(", block, re.MULTILINE):
        name = match.group(1)
        start = match.end()
        depth = 1
        index = start
        while depth:
            if block[index] == "(":
                depth += 1
            elif block[index] == ")":
                depth -= 1
            index += 1
        columns: dict[str, str] = {}
        for entry in _split_top_level(block[start : index - 1]):
            if _PK_LINE.search(entry):
                continue
            columns[entry.split()[0]] = entry
        tables[name] = columns
    return tables


def test_spec_block_is_readable() -> None:
    """Если разбор сломался, остальные проверки стали бы бессмысленно зелёными.

    Опорные имена взяты из начала, середины и конца §3, а не задано число:
    число пришлось бы помнить, и ровно на этом в WP-01 сломался
    `test_spec_table_is_readable` с его `>= 26`.
    """
    tables = _spec_tables()
    assert tables, "блок §3 не разобрался вовсе"
    for anchor in ("users", "facts", "llm_calls"):
        assert anchor in tables, f"разбор не дошёл до {anchor}"
    assert "tg_id" in tables["users"]
    assert "fragment_ids" in tables["facts"]


def test_all_spec_tables_declared() -> None:
    """Состав таблиц совпадает с §3 в обе стороны."""
    spec = set(_spec_tables())
    declared = set(Base.metadata.tables)
    assert spec - declared == set(), f"есть в §3, нет в моделях: {sorted(spec - declared)}"
    assert declared - spec == set(), f"есть в моделях, нет в §3: {sorted(declared - spec)}"


@pytest.mark.parametrize("table", sorted(_spec_tables()))
def test_all_spec_columns_declared(table: str) -> None:
    """Состав колонок каждой таблицы совпадает с §3 в обе стороны."""
    spec = set(_spec_tables()[table])
    declared = set(Base.metadata.tables[table].columns.keys())
    assert spec - declared == set(), f"{table}: есть в §3, нет в модели: {sorted(spec - declared)}"
    assert declared - spec == set(), f"{table}: есть в модели, нет в §3: {sorted(declared - spec)}"


def _spec_columns_marked(marker: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for table, columns in _spec_tables().items():
        for column, declaration in columns.items():
            if marker in declaration.upper():
                out.append((table, column))
    return out


def test_explicit_not_null_is_honoured() -> None:
    """Каждая колонка, помеченная в §3 как `NOT NULL`, обязательна в модели.

    Таких в §3 тринадцать. Числа здесь нет намеренно: список берётся из
    спецификации, и появление четырнадцатой пометки не потребует правки теста.
    """
    marked = _spec_columns_marked("NOT NULL")
    assert marked, "в §3 не нашлось ни одной пометки NOT NULL — сломался разбор"
    for table, column in marked:
        assert (
            not Base.metadata.tables[table].columns[column].nullable
        ), f"{table}.{column} помечена NOT NULL в §3, но в модели nullable"


def test_difficulty_of_question_is_not_null_with_default() -> None:
    """`questions.difficulty NOT NULL DEFAULT 2` — обе половины свойства.

    Правка v3.4, внесённая потому, что `difficulty <= cap` при `NULL` даёт
    `NULL`, и вопрос молча выпадает из выборки планировщика (§13.3). Проверять
    обязательность и не проверять default значило бы закрыть половину.
    """
    column = Base.metadata.tables["questions"].columns["difficulty"]
    assert not column.nullable
    assert column.server_default is not None
    assert "2" in str(column.server_default.arg)  # type: ignore[union-attr]


def test_foreign_keys_match_spec_marks() -> None:
    """`FK` в §3 означает внешний ключ в модели, и только там.

    В `llm_calls` пометки `FK` нет ни у `user_id`, ни у `material_id` — там
    ограничений нет. Это буквальное следование §3: добавить их по своей
    инициативе значило бы реализовать больше, чем требует спецификация (§40).
    """
    for table, column in _spec_columns_marked("FK"):
        assert (
            Base.metadata.tables[table].columns[column].foreign_keys
        ), f"{table}.{column} помечена FK в §3, но внешнего ключа в модели нет"

    for name in ("user_id", "material_id"):
        assert (
            not Base.metadata.tables["llm_calls"].columns[name].foreign_keys
        ), f"llm_calls.{name}: §3 не помечает её FK, ограничения быть не должно"


def test_nullable_foreign_keys_are_those_marked_null() -> None:
    """`FK NULL` — необязательна, `FK` без пометки — обязательна.

    Читаю §3 так: явная пометка `NULL` у части внешних ключей (`subject_id FK
    NULL`, `materials.material_id FK NULL` в `sessions`) различает их от
    остальных. Иначе пометка не несла бы смысла. Заполнение умолчания, не
    разрешение противоречия; записано в ADR-0004.
    """
    for table, column in _spec_columns_marked("FK"):
        declaration = _spec_tables()[table][column].upper()
        expected_nullable = bool(re.search(r"\bNULL\b", declaration)) and (
            "NOT NULL" not in declaration
        )
        actual = Base.metadata.tables[table].columns[column].nullable
        assert actual is expected_nullable, (
            f"{table}.{column}: в §3 «{_spec_tables()[table][column].strip()}», "
            f"ожидалось nullable={expected_nullable}, в модели {actual}"
        )


def test_review_states_has_composite_primary_key() -> None:
    """Составной PK `(user_id, fact_id)` — §3, и он единственный такой.

    Отсюда же берётся число индексов под первичные ключи в контракте схемы:
    четырнадцать, а не тринадцать.
    """
    pk = Base.metadata.tables["review_states"].primary_key
    assert [c.name for c in pk.columns] == ["user_id", "fact_id"]


def test_confidence_and_quality_are_real() -> None:
    """Оба множителя `final_confidence` хранятся в `REAL` (§3).

    От этого зависит тип параметра `teachable_facts(min_confidence real)`:
    сравнение в `double precision` даёт ложь на порогах 0.65, 0.7 и 0.9 из-за
    ошибки представления float4. Если тип колонки когда-нибудь изменится,
    подпись функции обязана измениться вместе с ним — этот тест держит связь.
    """
    quality = Base.metadata.tables["fragments"].columns["quality"]
    confidence = Base.metadata.tables["facts"].columns["confidence"]
    for column in (quality, confidence):
        assert column.type.__class__.__name__ == "REAL", column.type
