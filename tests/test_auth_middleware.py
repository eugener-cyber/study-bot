"""Белый список: свои проходят, чужие отклоняются молча (§1.3)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiogram.types import User

from bot.middlewares.auth import AuthMiddleware


def _user(user_id: int) -> User:
    return User(id=user_id, is_bot=False, first_name="T")


@pytest.mark.asyncio
async def test_allowed_user_passes() -> None:
    handler = AsyncMock(return_value="ok")
    middleware = AuthMiddleware([111, 222])
    result = await middleware(handler, object(), {"event_from_user": _user(111)})
    assert result == "ok"
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_user_is_dropped_silently() -> None:
    """Хендлер не вызван и ответа нет вовсе: любой ответ подтверждает бота."""
    handler = AsyncMock()
    event = AsyncMock()
    middleware = AuthMiddleware([111])
    result = await middleware(handler, event, {"event_from_user": _user(999)})

    assert result is None
    handler.assert_not_awaited()
    event.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_event_without_user_is_dropped() -> None:
    handler = AsyncMock()
    middleware = AuthMiddleware([111])
    assert await middleware(handler, object(), {}) is None
    handler.assert_not_awaited()
