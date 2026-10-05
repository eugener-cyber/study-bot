"""Усечение пользовательского содержимого до 80 символов (§31)."""

from __future__ import annotations

import logging

import structlog

from core.logging import (
    TRUNCATE_AT,
    TRUNCATED_FIELDS,
    configure_logging,
    get_logger,
    truncate_user_content,
)


def test_long_field_truncated_exactly() -> None:
    long_text = "я" * 500
    result = truncate_user_content(None, "info", {"event": "x", "text": long_text})
    assert len(result["text"]) == TRUNCATE_AT


def test_no_tail_kept_anywhere() -> None:
    """Хвост не переезжает в соседнее поле: цель — чтобы его не было в логе."""
    long_text = "абвгд" * 100
    result = truncate_user_content(None, "info", {"text": long_text})
    assert result["text"] == long_text[:TRUNCATE_AT]
    assert len(result) == 1


def test_short_field_untouched() -> None:
    result = truncate_user_content(None, "info", {"text": "коротко"})
    assert result["text"] == "коротко"


def test_service_fields_not_truncated() -> None:
    """Усекать error или stage вредно — они и нужны целиком."""
    long_error = "E" * 500
    result = truncate_user_content(None, "info", {"error": long_error})
    assert result["error"] == long_error


def test_truncated_fields_cover_user_content() -> None:
    for field in ("text", "statement", "answer", "question"):
        assert field in TRUNCATED_FIELDS


def test_get_logger_type_matches_annotation() -> None:
    """Аннотация `get_logger` обязана совпадать с фактическим объектом.

    Прежде она объявляла `structlog.stdlib.BoundLogger`, а `type: ignore`
    закрывал расхождение. В результате `mypy --strict` на `core/` — проверка,
    которая по §35 и существует для защиты логики — пропускал вызов
    `log.setLevel(10)`, падающий `AttributeError` в рантайме.
    """
    configure_logging()
    bound = get_logger("test").bind()

    assert isinstance(bound, structlog.make_filtering_bound_logger(logging.INFO))
    assert not isinstance(bound, structlog.stdlib.BoundLogger)
