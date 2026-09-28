"""Структурное логирование.

ТЗ §31: логи в structlog JSON с `user_id` и `material_id` в контексте; полный
текст учебных материалов в логи не попадает, максимум первые 80 символов там,
где это действительно необходимо.

Усечение сделано процессором, а не дисциплиной вызывающего: правило, которое
надо помнить на каждой строчке логирования, рано или поздно забудут.
"""

from __future__ import annotations

import logging

import structlog
from structlog.typing import EventDict, WrappedLogger

TRUNCATE_AT = 80
"""Предел длины для полей с пользовательским содержимым (§31)."""

TRUNCATED_FIELDS = frozenset(
    {
        "text",
        "excerpt",
        "statement",
        "answer",
        "answer_raw",
        "question",
        "caption",
        "material_text",
        "fragment_text",
    }
)
"""Поля, которые могут содержать учебный материал или ввод пользователя.

Перечень, а не «всё подряд»: усекать `error` или `stage` бессмысленно и вредно.
"""


def truncate_user_content(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Усекает помеченные поля до `TRUNCATE_AT` символов.

    Хвост не сохраняется нигде: цель §31 — чтобы учебный текст не оседал в логах,
    а не чтобы он там лежал в другом месте.
    """
    for key in TRUNCATED_FIELDS & event_dict.keys():
        value = event_dict[key]
        if isinstance(value, str) and len(value) > TRUNCATE_AT:
            event_dict[key] = value[:TRUNCATE_AT]
    return event_dict


def configure_logging(level: int = logging.INFO) -> None:
    """Настраивает structlog на JSON-вывод с усечением."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            truncate_user_content,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Логгер с именем модуля."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
