"""Глобальный перехват исключений.

ТЗ §31: необработанное исключение — пользователь получает безопасное сообщение,
трейс уходит в лог, процесс бота не падает.

Это **обработчик наблюдателя `errors`**, а не middleware. Так было не всегда:
до ревью PR #3 здесь стоял `BaseMiddleware`, зарегистрированный на обсерверах
`message` и `callback_query`. Такой middleware вызывается внутри
`TelegramEventObserver.trigger`, то есть после фильтров и после собственных
outer middleware диспетчера — `UserContextMiddleware` и `FSMContextMiddleware`.
Всё, что падало раньше, он не видел: Проверяющий уронил `redis.get`, и
пользователь получил тишину вместо `SAFE_REPLY`, хотя §31 требует сообщение.

Наблюдатель `errors` вызывается самым внешним middleware самого aiogram
(`aiogram.dispatcher.middlewares.error.ErrorsMiddleware`, который Dispatcher
регистрирует первым в `__init__`), поэтому покрывает FSM, резолвинг контекста,
фильтры и хендлеры разом.
"""

from __future__ import annotations

from aiogram.types import ErrorEvent

from bot.notify import notify
from core.logging import get_logger

log = get_logger(__name__)

SAFE_REPLY = "Что-то пошло не так. Ошибка записана, я разберусь."


async def handle_error(event: ErrorEvent) -> bool:
    """Логирует трейс и отвечает пользователю без подробностей.

    Возвращает `True` всегда: любое значение, кроме `UNHANDLED`, говорит
    aiogram, что ошибка обработана, и он не поднимает её заново. Исключение,
    поднятое повторно, пережило бы цикл поллинга — процесс бы выжил, но §31
    требует ещё и сообщения пользователю, а его в этом случае не будет.
    """
    # exc_info передаётся явно объектом исключения, а не через `log.exception`:
    # обработчик вызывается из блока `except` чужого кода, и опираться на
    # `sys.exc_info()` здесь значит зависеть от того, что aiogram не станет
    # вызывать нас иначе.
    log.error(
        "unhandled_exception",
        error=type(event.exception).__name__,
        update_id=event.update.update_id,
        exc_info=event.exception,
    )
    await notify(event.update, SAFE_REPLY, alert=True)
    return True
