"""Перехват исключений: пользователю безопасный текст, процесс жив (§31)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message

from bot.middlewares.errors import SAFE_REPLY, ErrorsMiddleware


def _message() -> Message:
    message = Message(
        message_id=1,
        date=__import__("datetime").datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
    )
    object.__setattr__(message, "answer", AsyncMock())
    return message


@pytest.mark.asyncio
async def test_exception_is_caught_and_user_notified() -> None:
    async def failing(_event: object, _data: dict[str, object]) -> None:
        raise RuntimeError("что-то сломалось")

    message = _message()
    result = await ErrorsMiddleware()(failing, message, {})

    assert result is None  # исключение не вылетело наружу
    message.answer.assert_awaited_once_with(SAFE_REPLY)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_safe_reply_has_no_traceback() -> None:
    """Пользователь не должен видеть внутренности (§31)."""
    assert "Traceback" not in SAFE_REPLY
    assert "Error" not in SAFE_REPLY


@pytest.mark.asyncio
async def test_failure_inside_notification_does_not_escape() -> None:
    """Сбой самого уведомления не должен ронять процесс."""

    async def failing(_event: object, _data: dict[str, object]) -> None:
        raise RuntimeError("первичная ошибка")

    message = _message()
    message.answer.side_effect = RuntimeError("и ответить тоже не вышло")  # type: ignore[attr-defined]

    assert await ErrorsMiddleware()(failing, message, {}) is None


@pytest.mark.asyncio
async def test_successful_handler_passes_through() -> None:
    handler = AsyncMock(return_value="ok")
    assert await ErrorsMiddleware()(handler, _message(), {}) == "ok"
