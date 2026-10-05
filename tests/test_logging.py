"""Усечение пользовательского содержимого до 80 символов (§31)."""

from __future__ import annotations

import logging

import structlog

from core.logging import (
    NEVER_TRUNCATED,
    TRUNCATE_AT,
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
    """Усекать имя стадии или класс ошибки вредно — они нужны целиком."""
    long_value = "E" * 500
    result = truncate_user_content(None, "info", {"stage": long_value})
    assert result["stage"] == long_value


def test_error_message_is_truncated() -> None:
    """Текст исключения — пользовательское содержимое, а не диагностика.

    До ревью фазы 2 поле `error` стояло в перечне исключений и сохранялось
    целиком. Сообщения исключений штатно содержат входные данные, а
    `error=str(exc)` — самая естественная запись в коде. Класс ошибки
    логируется как `error_type` и остаётся целым.
    """
    long_message = "invalid literal for int(): " + "9" * 300
    result = truncate_user_content(
        None, "info", {"error": long_message, "error_type": "ValueError"}
    )
    assert len(result["error"]) == TRUNCATE_AT
    assert result["error_type"] == "ValueError"


def test_limit_is_eighty_as_required_by_spec() -> None:
    """§31 называет число буквально: «максимум первые 80 символов».

    Остальные тесты выражены через `TRUNCATE_AT` и поэтому проходят при любом
    его значении — проверено, с 200 они тоже зелёные. Норму держит только этот
    ассерт, и он здесь единственный, где 80 стоит литералом.
    """
    assert TRUNCATE_AT == 80


def test_unlisted_field_is_truncated() -> None:
    """Любое неизвестное поле считается пользовательским содержимым.

    Это главное свойство схемы «запрещено по умолчанию»: с прежним перечнем
    усекаемых полей `answer_text` уезжал в лог целиком, потому что в список
    попал `answer`, а не `answer_text`.
    """
    long_text = "ю" * 300
    result = truncate_user_content(
        None,
        "info",
        {"answer_text": long_text, "question_text": long_text, "note": long_text},
    )
    for key in ("answer_text", "question_text", "note"):
        assert len(result[key]) == TRUNCATE_AT, f"{key} не усечено"


def test_service_fields_listed_explicitly() -> None:
    """Перечень исключений закрытый и содержит только служебные имена."""
    for field in ("event", "level", "timestamp", "error_type", "exception"):
        assert field in NEVER_TRUNCATED
    assert "error" not in NEVER_TRUNCATED, "текст исключения обязан усекаться"
    for field in ("text", "statement", "answer", "question", "material_text"):
        assert field not in NEVER_TRUNCATED


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
