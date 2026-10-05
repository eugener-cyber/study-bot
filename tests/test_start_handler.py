"""Команда /start и обработка согласия (§31, CR-B)."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram.types import CallbackQuery, Chat, Message, User

from bot.handlers.start import (
    CONSENT_ACCEPTED,
    CONSENT_CALLBACK,
    GREETING,
    consent_keyboard,
    handle_consent,
    handle_start,
)


def _message() -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
        from_user=User(id=111, is_bot=False, first_name="T"),
    )
    object.__setattr__(message, "answer", AsyncMock())
    object.__setattr__(message, "edit_text", AsyncMock())
    return message


@pytest.mark.asyncio
async def test_start_returns_consent_text() -> None:
    message = _message()
    await handle_start(message)
    message.answer.assert_awaited_once()  # type: ignore[attr-defined]
    assert message.answer.await_args.args[0] == GREETING  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_start_attaches_keyboard() -> None:
    message = _message()
    await handle_start(message)
    kwargs = message.answer.await_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == CONSENT_CALLBACK


@pytest.mark.asyncio
async def test_consent_click_is_handled_without_db() -> None:
    """Запись users.consent_at перенесена в WP-02 (CR-B); нажатие не падает."""
    inner = _message()
    callback = CallbackQuery(
        id="1",
        from_user=User(id=111, is_bot=False, first_name="T"),
        chat_instance="x",
        data=CONSENT_CALLBACK,
        message=inner,
    )
    object.__setattr__(callback, "answer", AsyncMock())

    await handle_consent(callback)

    callback.answer.assert_awaited_once()  # type: ignore[attr-defined]
    inner.edit_text.assert_awaited_once_with(CONSENT_ACCEPTED)  # type: ignore[attr-defined]


def test_consent_text_mentions_storage() -> None:
    """Согласие должно говорить, что именно хранится (§31)."""
    assert "матери" in GREETING and "хран" in GREETING


def test_todo_points_at_wp02_issue() -> None:
    """Отложенная запись согласия обязана иметь тикет и владельца (костыль п. 10)."""
    source = Path(__file__).resolve().parents[1] / "bot" / "handlers" / "start.py"
    text = source.read_text(encoding="utf-8")
    assert "TODO(#2, owner:" in text


def test_keyboard_has_single_button() -> None:
    keyboard = consent_keyboard().inline_keyboard
    assert len(keyboard) == 1 and len(keyboard[0]) == 1


def test_consent_does_not_invite_to_send_material() -> None:
    """§19.4: приглашать прислать материал нельзя, пока бот его не принимает.

    Текст согласия заканчивался словами «Пришлите первый материал», а
    обработчика для файлов не было — молчание после приглашения.
    """
    assert "ришлите" not in CONSENT_ACCEPTED
