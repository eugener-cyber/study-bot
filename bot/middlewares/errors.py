"""Глобальный перехват исключений.

ТЗ §31: необработанное исключение — пользователь получает безопасное сообщение,
трейс уходит в лог, процесс бота не падает.

Регистрируется первым, до auth и throttle: ошибка в любом последующем слое
всё равно доходит сюда.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from core.logging import get_logger

log = get_logger(__name__)

SAFE_REPLY = "Что-то пошло не так. Ошибка записана, я разберусь."


class ErrorsMiddleware(BaseMiddleware):
    """Ловит всё, что не поймали ниже."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except Exception:
            log.exception("unhandled_exception", event_type=type(event).__name__)
            await self._notify(event)
            return None

    @staticmethod
    async def _notify(event: TelegramObject) -> None:
        """Сообщает пользователю без подробностей.

        Сбой самого уведомления гасится: иначе исключение из обработчика ошибок
        вылетело бы наружу и уронило процесс — ровно то, что §31 запрещает.
        """
        try:
            if isinstance(event, Message):
                await event.answer(SAFE_REPLY)
            elif isinstance(event, CallbackQuery):
                await event.answer(SAFE_REPLY, show_alert=True)
        except Exception:
            log.exception("failed_to_notify_user")
