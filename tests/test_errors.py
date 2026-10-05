"""Перехват исключений: пользователю безопасный текст, процесс жив (§31)."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

from aiogram.types import CallbackQuery, Chat, ErrorEvent, Message, Update, User

from bot.errors import SAFE_REPLY, ErrorHandler, build_error_handler
from bot.notify import notify, respondable


def _message() -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
        from_user=User(id=111, is_bot=False, first_name="T"),
    )
    object.__setattr__(message, "answer", AsyncMock())
    return message


ALLOWED_ID = 111
OUTSIDER_ID = 999


def _error_event(update: Update) -> ErrorEvent:
    return ErrorEvent(update=update, exception=RuntimeError("что-то сломалось"))


def _handler(allowed: list[int] | None = None) -> ErrorHandler:
    return build_error_handler([ALLOWED_ID] if allowed is None else allowed)


async def test_user_is_notified_on_message_update() -> None:
    message = _message()
    assert await _handler()(_error_event(Update(update_id=1, message=message))) is True
    message.answer.assert_awaited_once_with(SAFE_REPLY)  # type: ignore[attr-defined]


async def test_user_is_notified_on_callback_update() -> None:
    callback = CallbackQuery(
        id="1",
        from_user=User(id=111, is_bot=False, first_name="T"),
        chat_instance="x",
        data="whatever",
    )
    object.__setattr__(callback, "answer", AsyncMock())

    assert await _handler()(_error_event(Update(update_id=1, callback_query=callback))) is True

    callback.answer.assert_awaited_once_with(SAFE_REPLY, show_alert=True)  # type: ignore[attr-defined]


async def test_returns_true_so_aiogram_does_not_reraise() -> None:
    """Любое значение, кроме UNHANDLED, означает «ошибка обработана».

    Если вернуть UNHANDLED, aiogram поднимет исключение заново: процесс
    выживет на цикле поллинга, но §31 требует ещё и сообщения пользователю,
    а его в этом случае уже не будет.
    """
    result = await _handler()(_error_event(Update(update_id=1, message=_message())))
    assert result is not None and result is not False


async def test_update_without_user_target_does_not_crash() -> None:
    """Апдейт, на который некому отвечать, не должен ронять обработчик."""
    assert await _handler()(_error_event(Update(update_id=1))) is True


async def test_failure_inside_notification_does_not_escape() -> None:
    """Сбой самого уведомления не должен ронять процесс."""
    message = _message()
    message.answer.side_effect = RuntimeError("и ответить тоже не вышло")  # type: ignore[attr-defined]

    assert await _handler()(_error_event(Update(update_id=1, message=message))) is True


async def test_outsider_gets_no_safe_reply() -> None:
    """Постороннему перехватчик не отвечает (§1.3).

    Наблюдатель `errors` стоит снаружи цепочки middleware, то есть снаружи
    `AuthMiddleware`: без проверки здесь сбой до авторизации давал
    постороннему ответ. Молчание, а не отказ — по той же причине, что в
    `bot/middlewares/auth.py`: любой ответ подтверждает, что бот живой.
    """
    message = _message()
    object.__setattr__(message.from_user, "id", OUTSIDER_ID)

    assert await _handler()(_error_event(Update(update_id=1, message=message))) is True

    message.answer.assert_not_awaited()  # type: ignore[attr-defined]


async def test_update_without_sender_gets_no_reply() -> None:
    """Не удалось определить отправителя — не отвечаем.

    Отказ по умолчанию: апдейт без пользователя (например, `channel_post`)
    не должен получать ответ лишь потому, что проверить его не удалось.
    """
    message = _message()
    object.__setattr__(message, "from_user", None)

    assert await _handler()(_error_event(Update(update_id=1, message=message))) is True

    message.answer.assert_not_awaited()  # type: ignore[attr-defined]


async def test_empty_whitelist_answers_nobody() -> None:
    """Пустой белый список — граничный случай: не отвечаем никому."""
    message = _message()

    assert await _handler(allowed=[])(_error_event(Update(update_id=1, message=message))) is True

    message.answer.assert_not_awaited()  # type: ignore[attr-defined]


def test_safe_reply_has_no_traceback() -> None:
    """Пользователь не должен видеть внутренности (§31)."""
    assert "Traceback" not in SAFE_REPLY
    assert "Error" not in SAFE_REPLY


def test_respondable_unwraps_update() -> None:
    message = _message()
    assert respondable(Update(update_id=1, message=message)) is message


def test_respondable_returns_none_when_nowhere_to_answer() -> None:
    """None — нормальный исход: на my_chat_member отвечать некому."""
    assert respondable(Update(update_id=1)) is None


async def test_notify_reports_failure_instead_of_raising() -> None:
    message = _message()
    message.answer.side_effect = RuntimeError("не вышло")  # type: ignore[attr-defined]
    assert await notify(Update(update_id=1, message=message), "текст") is False
