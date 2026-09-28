"""Троттлинг: норма §30.3, счётчики в Redis, одно предупреждение за окно."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.types import User

from bot.middlewares.throttle import WARNING, ThrottleMiddleware


def _redis(counter_value: int) -> MagicMock:
    """Redis, чей pipeline возвращает заданное значение счётчика."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[counter_value, True])
    redis = MagicMock()
    redis.pipeline.return_value = pipe
    return redis


def _data() -> dict[str, object]:
    return {"event_from_user": User(id=111, is_bot=False, first_name="T")}


@pytest.mark.asyncio
async def test_under_limit_passes() -> None:
    handler = AsyncMock(return_value="ok")
    middleware = ThrottleMiddleware(_redis(5), limit=20, window_sec=60)
    assert await middleware(handler, AsyncMock(), _data()) == "ok"


@pytest.mark.asyncio
async def test_at_limit_still_passes() -> None:
    """Граница включительна: двадцатое сообщение из двадцати разрешённых."""
    handler = AsyncMock(return_value="ok")
    middleware = ThrottleMiddleware(_redis(20), limit=20, window_sec=60)
    assert await middleware(handler, AsyncMock(), _data()) == "ok"


@pytest.mark.asyncio
async def test_first_excess_warns_once() -> None:
    from aiogram.types import Chat, Message

    handler = AsyncMock()
    message = Message(
        message_id=1,
        date=__import__("datetime").datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
    )
    object.__setattr__(message, "answer", AsyncMock())

    middleware = ThrottleMiddleware(_redis(21), limit=20, window_sec=60)
    assert await middleware(handler, message, _data()) is None
    handler.assert_not_awaited()
    message.answer.assert_awaited_once_with(WARNING)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_further_excess_is_silent() -> None:
    """Зажатая кнопка не должна превращаться в поток предупреждений."""
    handler = AsyncMock()
    event = AsyncMock()
    middleware = ThrottleMiddleware(_redis(50), limit=20, window_sec=60)

    assert await middleware(handler, event, _data()) is None
    handler.assert_not_awaited()
    event.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_ttl_set_only_on_creation() -> None:
    """Иначе окно сдвигалось бы с каждым сообщением и не истекало никогда."""
    redis = _redis(1)
    middleware = ThrottleMiddleware(redis, limit=20, window_sec=60)
    await middleware(AsyncMock(), AsyncMock(), _data())
    redis.pipeline.return_value.expire.assert_called_once_with("throttle:111", 60, nx=True)


@pytest.mark.asyncio
async def test_event_without_user_passes() -> None:
    handler = AsyncMock(return_value="ok")
    middleware = ThrottleMiddleware(_redis(1), limit=20, window_sec=60)
    assert await middleware(handler, AsyncMock(), {}) == "ok"
