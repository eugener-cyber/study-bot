"""Структурное логирование.

ТЗ §31: логи в structlog JSON с `user_id` и `material_id` в контексте; полный
текст учебных материалов в логи не попадает, максимум первые 80 символов там,
где это действительно необходимо.

Усечение сделано процессором, а не дисциплиной вызывающего: правило, которое
надо помнить на каждой строчке логирования, рано или поздно забудут.

**Запрещено по умолчанию, разрешено по списку.** Усекается любое строковое
значение длиннее `TRUNCATE_AT`; целиком сохраняются только поля из
`NEVER_TRUNCATED`. Обратная схема — перечень полей, которые нужно усекать —
стояла здесь до ревью PR #3 и дисциплину вызывающего не убирала, а лишь
меняла её формулировку: вместо «не забудь обрезать» требовалось «не забудь
назвать поле ровно одной из девяти строк». Поле `answer_text` вместо `answer`
уезжало в лог целиком, и ни тест, ни линтер этого не замечали.
"""

from __future__ import annotations

import logging
from typing import cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

TRUNCATE_AT = 80
"""Предел длины строковых значений в логе.

Число задано §31 буквально: «максимум первые 80 символов». Менять его здесь
нельзя без правки ТЗ — это проверяет `tests/test_logging.py`.
"""

NEVER_TRUNCATED = frozenset(
    {
        # Служебные поля самого structlog и процессоров ниже по цепочке.
        "event",
        "level",
        "logger",
        "logger_name",
        "timestamp",
        "exception",
        "exc_info",
        "stack",
        # Диагностика: усечённый трейсбек или класс ошибки бесполезны.
        "error",
        "stage",
        "event_type",
    }
)
"""Поля, которые сохраняются целиком.

Короткий закрытый перечень служебных имён. Всё, что в него не входит, считается
потенциальным пользовательским содержимым и усекается. Добавление имени сюда —
осознанное решение о том, что по этому полю учебный текст не пойдёт.
"""


def truncate_user_content(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Усекает все строковые значения до `TRUNCATE_AT`, кроме служебных.

    Хвост не сохраняется нигде: цель §31 — чтобы учебный текст не оседал в логах,
    а не чтобы он там лежал в другом месте.

    Вложенные структуры не обходятся: словарь или список в логе — это уже
    нарушение §31 независимо от длины, и подменять его усечением значило бы
    делать нарушение незаметным. Такие значения остаются как есть и видны.
    """
    for key, value in event_dict.items():
        if key in NEVER_TRUNCATED:
            continue
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
