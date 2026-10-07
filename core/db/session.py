"""Транзакционная граница.

Перенос из WP-01 (Issue #2, блок «Перенесено из WP-01»): ТЗ §35 — «доступ к БД
только через `core/services`, сессия прокидывается middleware».

Граница одна и явная: `session_scope` открывает сессию, коммитит при успешном
выходе и откатывает при любом исключении. Чек-лист Проверяющего, раздел 2,
требует, чтобы частичный результат при сбое посередине был невозможен; с
коммитом, раскиданным по сервисам, это свойство нельзя ни прочитать, ни
проверить.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Сессия с коммитом на выходе и откатом при исключении.

    Исключение не гасится: откат делается и исключение поднимается дальше, к
    перехватчику `bot/errors.py`. Проглоченная здесь ошибка означала бы, что
    пользователь получил ответ об успехе при незаписанных данных.
    """
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            log.error("db_transaction_rolled_back", exc_info=True)
            raise
