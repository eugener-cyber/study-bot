"""Структурное логирование.

ТЗ §31: логи в structlog JSON с `user_id` и `material_id` в контексте; полный
текст учебных материалов в логи не попадает, максимум первые 80 символов там,
где это действительно необходимо.

Усечение сделано процессором, а не дисциплиной вызывающего: правило, которое
надо помнить на каждой строчке логирования, рано или поздно забудут.
"""

from __future__ import annotations

import logging
from typing import cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

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


def get_logger(name: str) -> FilteringBoundLogger:
    """Логгер с именем модуля.

    Тип возврата — `FilteringBoundLogger`, а не `structlog.stdlib.BoundLogger`:
    `configure_logging` ставит `wrapper_class=make_filtering_bound_logger(...)`,
    и фактический объект — `BoundLoggerLazyProxy`, после первого использования
    `BoundLoggerFilteringAtInfo`. Методов `stdlib.BoundLogger` (`setLevel`,
    `addHandler`) у него нет.

    Прежняя аннотация вместе с `type: ignore[no-any-return]` заставляла
    `mypy --strict` верить в несуществующий тип: вызов `log.setLevel(10)`
    проходил проверку и падал `AttributeError` в рантайме. `cast` вместо
    `type: ignore` оставляет проверку рабочей — `structlog.get_logger`
    не типизирован и возвращает `Any`, и это единственное, что здесь
    действительно нужно подавить.
    """
    return cast(FilteringBoundLogger, structlog.get_logger(name))
