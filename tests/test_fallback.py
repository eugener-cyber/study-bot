"""Ответ на неподдерживаемый ввод (§19.4, переходное требование)."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, PhotoSize, User

from bot.handlers.fallback import REPLY, _content_kind, _may_reply, build_fallback_router


def _message(**extra: object) -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
        from_user=User(id=111, is_bot=False, first_name="T"),
        **extra,  # type: ignore[arg-type]
    )
    object.__setattr__(message, "answer", AsyncMock())
    return message


def _redis(first_time: bool) -> AsyncMock:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True if first_time else None)
    return redis


async def _handle(router, message: Message) -> None:  # type: ignore[no-untyped-def]
    """Достаёт единственный хендлер роутера и зовёт его напрямую."""
    handler = router.message.handlers[0].callback
    await handler(message)


@pytest.mark.asyncio
async def test_first_unsupported_message_gets_reply() -> None:
    router = build_fallback_router(_redis(True), cooldown_sec=60)
    message = _message(text="привет")
    await _handle(router, message)
    message.answer.assert_awaited_once_with(REPLY)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_second_message_within_cooldown_is_silent() -> None:
    """Альбом из десяти фото не должен дать десять одинаковых ответов."""
    router = build_fallback_router(_redis(False), cooldown_sec=60)
    message = _message(text="ещё")
    await _handle(router, message)
    message.answer.assert_not_awaited()  # type: ignore[attr-defined]


def test_reply_says_file_not_saved() -> None:
    """Без этого пользователь решит, что файл в очереди, и будет ждать (§19.4)."""
    assert "не сохранён" in REPLY


def test_reply_does_not_invite_to_send_again() -> None:
    assert "присылать заново не нужно" in REPLY


def test_content_kind_detects_photo() -> None:
    photo = [PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)]
    assert _content_kind(_message(photo=photo)) == "photo"


def test_content_kind_detects_text() -> None:
    assert _content_kind(_message(text="привет")) == "text"


@pytest.mark.asyncio
async def test_cooldown_key_set_with_nx() -> None:
    """NX: окно отсчитывается от первого ответа и не сдвигается следующими."""
    redis = _redis(True)
    assert await _may_reply(redis, 111, 60) is True
    redis.set.assert_awaited_once_with("unsupported:111", "1", ex=60, nx=True)


@pytest.mark.asyncio
async def test_cooldown_blocks_when_key_exists() -> None:
    assert await _may_reply(_redis(False), 111, 60) is False
