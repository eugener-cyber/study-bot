"""Сессия базы данных в хендлеры.

Перенос из WP-01 (Issue #2, блок «Перенесено из WP-01»): ТЗ §35 — «доступ к БД
только через `core/services`, сессия прокидывается middleware».

Регистрируется **после** авторизации и троттлинга (`bot/main.py`): иначе
обращение постороннего открывало бы транзакцию, а при частых обращениях —
занимало бы соединения из пула. Посторонний отклоняется раньше, чем сессия
создаётся.

Транзакционная граница живёт не здесь, а в `core/db/session.py`: middleware
только отдаёт сессию в `data`, а решение о коммите и откате принимает один
контекст на весь проект.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.db.session import session_scope

SESSION_KEY = "session"
"""Имя ключа в `data`. Хендлер объявляет параметр `session` — aiogram
подставляет его по имени."""


class DbSessionMiddleware(BaseMiddleware):
    """Открывает сессию на один апдейт и закрывает её вместе с ним."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        async with session_scope(self._sessions) as session:
            data[SESSION_KEY] = session
            return await handler(event, data)
