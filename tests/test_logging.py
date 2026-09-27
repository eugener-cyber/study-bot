"""Усечение пользовательского содержимого до 80 символов (§31)."""

from __future__ import annotations

from core.logging import TRUNCATE_AT, TRUNCATED_FIELDS, truncate_user_content


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
