"""Сообщение прогресса. ТЗ §4.5.

Класс существовал с прошлого коммита и **не был подключён ни к чему**: 105
строк мёртвого кода без единого теста. Нашло ревью PR #14. Здесь он
проверяется, а в `upload.py` и `worker.py` — используется.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import EditMessageText
from aiogram.types import Chat, Message

from bot.progress import MIN_INTERVAL_SEC, ProgressMessage, stage_text


def _message() -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
    )
    object.__setattr__(message, "edit_text", AsyncMock())
    return message


def _not_modified() -> TelegramBadRequest:
    return TelegramBadRequest(
        method=EditMessageText(text="x", chat_id=1, message_id=1),
        message=(
            "Bad Request: message is not modified: specified new message content and "
            "reply markup are exactly the same as a current content and reply markup "
            "of the message"
        ),
    )


def test_interval_matches_spec() -> None:
    """§4.5 называет интервал буквально: не чаще раза в 5 секунд.

    Число стоит литералом здесь и только здесь. Остальные тесты выражены
    через константу и прошли бы при любом её значении — тот же урок, что с
    `TRUNCATE_AT` в WP-01.
    """
    assert MIN_INTERVAL_SEC == 5.0


async def test_first_update_goes_through() -> None:
    message = _message()
    progress = ProgressMessage(message)

    assert await progress.update("⏳ Разбираю материал…") is True
    message.edit_text.assert_awaited_once_with("⏳ Разбираю материал…")  # type: ignore[attr-defined]


async def test_second_update_within_interval_is_skipped() -> None:
    """Ограничитель частоты работает — то, ради чего класс и написан.

    `edit_message_text` чаще упирается в лимиты Telegram, и ответ приходит не
    отказом, а `TelegramRetryAfter`, то есть задержкой всего обработчика. При
    стадии, рапортующей о каждой странице трёхсотстраничного скана, обработка
    ждала бы Telegram дольше, чем работала.
    """
    message = _message()
    progress = ProgressMessage(message)

    await progress.update("первое")
    assert await progress.update("второе") is False
    assert progress.skipped == 1
    assert message.edit_text.await_count == 1  # type: ignore[attr-defined]


async def test_update_after_interval_goes_through() -> None:
    """Обратное направление: по истечении интервала правка проходит.

    Без этого теста проверка ограничителя прошла бы и у класса, который не
    правит сообщение никогда.
    """
    message = _message()
    progress = ProgressMessage(message, min_interval=0.0)

    await progress.update("первое")
    assert await progress.update("второе") is True
    assert message.edit_text.await_count == 2  # type: ignore[attr-defined]


async def test_force_ignores_the_interval() -> None:
    """Итоговое состояние доходит независимо от частоты.

    Без `force` пользователь мог бы остаться с текстом «обрабатываю» на
    завершённом материале — потому что последняя правка пришла слишком скоро
    после предыдущей.
    """
    message = _message()
    progress = ProgressMessage(message)

    await progress.update("обрабатываю")
    assert await progress.update("✅ Готово", force=True) is True
    assert message.edit_text.await_count == 2  # type: ignore[attr-defined]


async def test_same_text_is_not_resent() -> None:
    """Одинаковый текст не отправляется повторно.

    Две стадии подряд могут дать одну строку прогресса, и Telegram на такую
    правку отвечает ошибкой, а не молча.
    """
    message = _message()
    progress = ProgressMessage(message, min_interval=0.0)

    await progress.update("одно и то же")
    assert await progress.update("одно и то же") is False
    assert message.edit_text.await_count == 1  # type: ignore[attr-defined]


async def test_not_modified_is_swallowed() -> None:
    """«message is not modified» не роняет обработчик.

    Достижимо штатно: `force=True` обходит проверку текста, и Telegram тогда
    отвечает ошибкой. Падение здесь уронило бы обработку материала из-за
    косметики.
    """
    message = _message()
    message.edit_text.side_effect = _not_modified()  # type: ignore[attr-defined]
    progress = ProgressMessage(message)

    assert await progress.update("текст") is False


async def test_other_bad_request_is_not_swallowed() -> None:
    """Гасится ровно один случай, а не любая ошибка правки."""
    message = _message()
    message.edit_text.side_effect = TelegramBadRequest(  # type: ignore[attr-defined]
        method=EditMessageText(text="x", chat_id=1, message_id=1),
        message="Bad Request: message to edit not found",
    )
    progress = ProgressMessage(message)

    with pytest.raises(TelegramBadRequest):
        await progress.update("текст")


def test_stage_text_speaks_in_first_person() -> None:
    """Формулировки как в примере §4.5 — «Распознаю аудио…».

    Пользователь читает их как отчёт о том, что происходит сейчас, а не как
    название этапа в системе.
    """
    assert stage_text("extract") == "⏳ Разбираю материал…"
    assert stage_text("questions") == "⏳ Составляю задания…"


def test_stage_text_shows_counters_when_given() -> None:
    """§4.5: «⏳ Распознаю аудио… 4:12 / 58:30»."""
    assert stage_text("extract", 4, 58) == "⏳ Разбираю материал… 4 / 58"


def test_unknown_stage_falls_back_to_its_name() -> None:
    """Неизвестная стадия не роняет рендер.

    Перечень стадий §4.1 может пополниться (аудио в WP-19, видео в WP-20), и
    забытая строка в словаре не должна ломать сообщение прогресса.
    """
    assert "probe-2" in stage_text("probe-2")
