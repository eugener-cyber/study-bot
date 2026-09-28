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

# Значения, которые в ТЗ не числа и потому сверяются отдельно.
NON_NUMERIC = {"MAX_SECTION_TOKENS", "COST_WARN_THRESHOLD", "COST_CURRENCY"}


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


def test_spec_table_is_readable() -> None:
    """Если разбор сломался, остальные проверки стали бы бессмысленно зелёными."""
    assert len(_spec_thresholds()) >= 26


def test_no_threshold_missing_from_config() -> None:
    """В ТЗ появилась строка — она обязана появиться в Settings."""
    fields = set(Settings.model_fields)
    missing = {
        RENAMED.get(name, name)
        for name in _spec_thresholds()
        if RENAMED.get(name, name) not in fields
    }
    assert not missing, f"Пороги есть в §30.3, но нет в конфиге: {sorted(missing)}"


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
