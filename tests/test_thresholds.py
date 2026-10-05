"""Пороги §30.3 сверяются с таблицей ТЗ построчно.

Числа переписываются из таблицы руками, а используются начиная с WP-07.
Опечатка вида 0.55 -> 0.055 всплыла бы через несколько пакетов в виде «модель
почему-то пропускает мусор». Тест падает и на расхождении значения, и на
появлении в ТЗ строки, которой нет в конфиге, и наоборот.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.config import Settings

SPEC = Path(__file__).resolve().parents[1] / "docs" / "ТЗ.md"

# Имена, у которых в конфиге суффикс единиц: в ТЗ «90 мин», «10 мин», «24».
RENAMED = {
    "MIN_PUSH_INTERVAL": "MIN_PUSH_INTERVAL_MIN",
    "NEW_FACT_DELAY": "NEW_FACT_DELAY_MIN",
}

# Значения, которые в ТЗ действительно не числа.
# `COST_WARN_THRESHOLD` — `TODO(owner)`, выводится после спайка (§34.1).
# `COST_CURRENCY` — `USD` на этапе 1, `RUB` в эксплуатации.
# `MAX_SECTION_TOKENS` стоял здесь ошибочно: в ТЗ «60000 на claude-opus-5
# (окно 1M), уточняется при смене провайдера» — число есть, и разбор берёт
# его первым. Самый дорогой по последствиям порог §30.3 был исключён из
# сверки без причины (ревью PR #3, FAIL 3.3).
NON_NUMERIC = {"COST_WARN_THRESHOLD", "COST_CURRENCY"}


def _spec_thresholds() -> dict[str, str]:
    """Вытаскивает таблицу §30.3 из канонической копии ТЗ."""
    text = SPEC.read_text(encoding="utf-8")
    block = text.split("## 30.3")[1].split("\n# ")[0]
    found: dict[str, str] = {}
    for line in block.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        names = re.findall(r"`([A-Z_]{3,})`", cells[0])
        values = re.findall(r"\d+(?:[.,]\d+)?", cells[1])
        if len(names) > 1 and len(names) == len(values):
            # Парные строки вида «FAST_MS / SLOW_MS | 8000 / 45000»:
            # значения разносятся позиционно, а не дублируются.
            for name, value in zip(names, values, strict=True):
                found[name] = value
        else:
            for name in names:
                found[name] = cells[1]
    return found


def _spec_row_names() -> list[str]:
    """Имена из первой колонки §30.3 — независимо от разбора значений.

    Намеренно проще `_spec_thresholds`: та разносит парные строки по значениям
    и может их потерять, а эта только перечисляет имена.
    """
    text = SPEC.read_text(encoding="utf-8")
    block = text.split("## 30.3")[1].split("\n# ")[0]
    names: list[str] = []
    for line in block.splitlines():
        if line.startswith("| `"):
            names += re.findall(r"`([A-Z_]{3,})`", line.strip("|").split("|")[0])
    return names


def _spec_required_names() -> set[str]:
    """Обязательные переменные §30.2 из блока кода."""
    text = SPEC.read_text(encoding="utf-8")
    block = text.split("## 30.2")[1].split("## 30.3")[0]
    fenced = block.split("```")[1]
    return set(re.findall(r"\b([A-Z][A-Z_]{2,})\b", fenced))


def test_spec_table_is_readable() -> None:
    """Если разбор сломался, остальные проверки стали бы бессмысленно зелёными.

    Здесь стояло `>= 26`, тогда как разбор видит 30: потеря четырёх строк
    оставляла тест зелёным (ревью PR #3). Число больше не берётся из памяти о
    прошлой редакции — оно сверяется с независимым разбором первой колонки.
    """
    parsed = _spec_thresholds()
    assert parsed, "таблица §30.3 не разобралась вовсе"
    assert sorted(parsed) == sorted(
        _spec_row_names()
    ), "разбор значений потерял или добавил имена относительно первой колонки"
    # Непустота не доказывает, что разбор дошёл до конца таблицы: три опорных
    # имени взяты из начала, середины и конца §30.3.
    for anchor in ("MIN_FACT_CONFIDENCE", "MIN_PUSH_INTERVAL", "LANG_CONFIDENCE_MIN"):
        assert anchor in parsed, f"разбор не дошёл до {anchor}"


def test_no_threshold_missing_from_config() -> None:
    """В ТЗ появилась строка — она обязана появиться в Settings."""
    fields = set(Settings.model_fields)
    missing = {
        RENAMED.get(name, name)
        for name in _spec_thresholds()
        if RENAMED.get(name, name) not in fields
    }
    assert not missing, f"Пороги есть в §30.3, но нет в конфиге: {sorted(missing)}"


def test_no_config_field_missing_from_spec() -> None:
    """Обратное направление: поле появилось в конфиге — оно обязано быть в ТЗ.

    План обещал проверку «или наоборот», но реализовано было только одно
    направление: Проверяющий добавил в Settings поле `INVENTED_THRESHOLD`, и
    прогон остался зелёным (ревью PR #3, FAIL 4.4). Смысл направления виден
    на этом же пакете: `MAX_INTERVAL_DAYS` и `UNSUPPORTED_REPLY_COOLDOWN_SEC`
    пришли через change request, и эта проверка не дала бы добавить порог в
    конфиг, забыв про спецификацию.
    """
    documented = {RENAMED.get(name, name) for name in _spec_thresholds()}
    documented |= _spec_required_names()
    extra = set(Settings.model_fields) - documented
    assert not extra, f"Поля есть в конфиге, но не описаны в §30.2 и §30.3: {sorted(extra)}"


@pytest.mark.parametrize("name,raw", sorted(_spec_thresholds().items()))
def test_threshold_value_matches_spec(name: str, raw: str) -> None:
    """Значение по умолчанию совпадает с таблицей ТЗ."""
    field_name = RENAMED.get(name, name)
    if name in NON_NUMERIC:
        pytest.skip(f"{name}: значение в ТЗ не числовое ({raw})")

    numbers = re.findall(r"\d+(?:\.\d+)?", raw)
    assert numbers, f"{name}: в ТЗ нет числа ({raw})"
    expected = float(numbers[0])
    actual = float(Settings.model_fields[field_name].default)  # type: ignore[arg-type]
    assert actual == expected, f"{name}: в ТЗ {expected}, в конфиге {actual}"
