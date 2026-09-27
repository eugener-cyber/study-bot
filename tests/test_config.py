"""Конфигурация: обязательные переменные, разбор белого списка, пороги §30.3."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.config import Settings


def _settings(**overrides: str) -> Settings:
    base = {
        "BOT_TOKEN": "0000000000:X",
        "TELEGRAM_API_ID": "1",
        "TELEGRAM_API_HASH": "h" * 32,
        "ALLOWED_USER_IDS": "111,222",
        "DATABASE_URL": "postgresql+asyncpg://t:t@localhost/t",
        "REDIS_URL": "redis://localhost:6379/0",
        "TZ_DEFAULT": "Europe/Moscow",
        "LLM_PROVIDER": "manual",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_missing_required_variable_names_it() -> None:
    """Отсутствие обязательной переменной валит старт и называет её имя."""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, BOT_TOKEN="x")  # type: ignore[call-arg]
    assert "TELEGRAM_API_ID" in str(exc.value)


def test_allowed_user_ids_parsed() -> None:
    assert _settings(ALLOWED_USER_IDS="111,222").ALLOWED_USER_IDS == [111, 222]


def test_allowed_user_ids_with_spaces() -> None:
    assert _settings(ALLOWED_USER_IDS=" 111 , 222 ").ALLOWED_USER_IDS == [111, 222]


def test_allowed_user_ids_single() -> None:
    assert _settings(ALLOWED_USER_IDS="111").ALLOWED_USER_IDS == [111]


def test_allowed_user_ids_empty_rejected() -> None:
    """Пустой белый список недопустим: «открыт всем» — не вариант (§1.3)."""
    with pytest.raises(ValidationError) as exc:
        _settings(ALLOWED_USER_IDS="")
    assert "ALLOWED_USER_IDS" in str(exc.value)


def test_cost_warn_threshold_has_no_default() -> None:
    """В §30.3 у него TODO(owner); значение требуется с WP-03, не раньше."""
    assert _settings().COST_WARN_THRESHOLD is None


def test_cost_warn_threshold_is_decimal() -> None:
    """§6.3 сравнивает порог с денежной оценкой, в §3 стоимость NUMERIC."""
    assert isinstance(_settings(COST_WARN_THRESHOLD="10.50").COST_WARN_THRESHOLD, Decimal)
