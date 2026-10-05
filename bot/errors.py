"""Глобальный перехват исключений.

ТЗ §31: необработанное исключение — пользователь получает безопасное сообщение,
трейс уходит в лог, процесс бота не падает.

Это **обработчик наблюдателя `errors`**, а не middleware. Так было не всегда:
до ревью PR #3 здесь стоял `BaseMiddleware`, зарегистрированный на обсерверах
`message` и `callback_query`. Такой middleware вызывается внутри
`TelegramEventObserver.trigger`, то есть после фильтров и после собственных
outer middleware диспетчера — `UserContextMiddleware` и `FSMContextMiddleware`.
Всё, что падало раньше, он не видел: сбой Redis при резолвинге FSM давал
пользователю тишину вместо `SAFE_REPLY`.

Наблюдатель `errors` вызывается самым внешним middleware самого aiogram
(`aiogram.dispatcher.middlewares.error.ErrorsMiddleware`, который Dispatcher
регистрирует первым в `__init__`), поэтому покрывает FSM, резолвинг контекста,
фильтры и хендлеры разом.

**Белый список проверяется здесь повторно, и это не дублирование.** `errors`
стоит снаружи цепочки middleware, то есть снаружи `AuthMiddleware`, который
Dispatcher получает последним. Исключение, возникшее до авторизации, доходит
до обработчика ошибок с апдейтом любого отправителя, и без проверки бот
отвечал бы постороннему — ровно то, что §1.3 запрещает, и ровно в том
единственном сценарии, который посторонний может вызвать снаружи
(перезапуск Redis). «Пользователь» в §31 — это пользователь из белого
списка: постороннего бот пользователем не считает, §1.3 его отклоняет.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from aiogram.dispatcher.middlewares.user_context import UserContextMiddleware
from aiogram.types import ErrorEvent, User

from bot.notify import notify
from core.logging import get_logger

log = get_logger(__name__)

SAFE_REPLY = "Что-то пошло не так. Ошибка записана, я разберусь."

ErrorHandler = Callable[[ErrorEvent], Awaitable[bool]]


def _sender(event: ErrorEvent) -> User | None:
    """Отправитель апдейта, или `None`, если его нет.

    Используется разбор самого aiogram, а не собственный: типов апдейтов
    больше двадцати, и свой разбор отстал бы от библиотеки молча — как раз в
    сторону «не нашли пользователя, значит отвечаем всем». `data` с ключом
    `event_from_user` для этого не годится: он заполняется
    `UserContextMiddleware`, которая стоит **после** перехватчика ошибок, и
    при сбое внутри неё самой ключа ещё нет.
    """
    return UserContextMiddleware.resolve_event_context(event=event.update).user


def build_error_handler(allowed_user_ids: list[int]) -> ErrorHandler:
    """Обработчик ошибок, отвечающий только пользователям из белого списка.

    Фабрика, а не функция уровня модуля: белый список приходит из настроек, и
    замыкание держит его так же, как `AuthMiddleware` — своим полем.
    """
    allowed = frozenset(allowed_user_ids)

    async def handle_error(event: ErrorEvent) -> bool:
        """Логирует трейс и отвечает пользователю без подробностей.

        Возвращает `True` всегда: любое значение, кроме `UNHANDLED`, говорит
        aiogram, что ошибка обработана, и он не поднимает её заново.
        Исключение, поднятое повторно, пережило бы цикл поллинга — процесс бы
        выжил, но §31 требует ещё и сообщения пользователю, а его в этом
        случае не будет. Возврат `True` при подавленном ответе — то же самое:
        ошибка обработана, просто отвечать некому.
        """
        user = _sender(event)

        # exc_info передаётся явно объектом исключения, а не через
        # `log.exception`: обработчик вызывается из блока `except` чужого кода,
        # и опираться на `sys.exc_info()` здесь значит зависеть от того, что
        # aiogram не станет вызывать нас иначе.
        log.error(
            "unhandled_exception",
            error_type=type(event.exception).__name__,
            update_id=event.update.update_id,
            user_id=user.id if user else None,
            exc_info=event.exception,
        )

        if user is None or user.id not in allowed:
            # Молчание, а не отказ: см. `bot/middlewares/auth.py`. Ответ
            # подтверждает постороннему, что бот живой.
            log.info("error_reply_suppressed", user_id=user.id if user else None)
            return True

        await notify(event.update, SAFE_REPLY, alert=True)
        return True

    return handle_error
