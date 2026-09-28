"""Белый список пользователей.

ТЗ §1.3: доступ — белый список `ALLOWED_USER_IDS`, middleware отклоняет остальных.

Отклонение молчаливое. Любой ответ — даже «вам сюда нельзя» — подтверждает, что
бот существует и работает, и приглашает пробовать дальше. Молчание не даёт
постороннему ничего.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

from core.logging import get_logger

log = get_logger(__name__)


class AuthMiddleware(BaseMiddleware):
    """Пропускает только тех, чей id есть в белом списке."""

    def __init__(self, allowed_user_ids: list[int]) -> None:
        self._allowed = frozenset(allowed_user_ids)

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")

        if user is None or user.id not in self._allowed:
            # Ни ответа, ни подтверждения — см. docstring модуля.
            log.info("rejected_unknown_user", user_id=getattr(user, "id", None))
            return None

        return await handler(event, data)
