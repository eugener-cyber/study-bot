"""Ответ на неподдерживаемый ввод.

ТЗ §19.4 — **переходное требование**. Описывает поведение на пакетах, где бот
уже отвечает, но ещё не умеет принимать материалы. Вместе с WP-03 этот модуль,
раздел §19.4 и константа `UNSUPPORTED_REPLY_COOLDOWN_SEC` удаляются.

Роутер регистрируется последним: он ловит всё, что не разобрали хендлеры выше.
Файл не скачивается и никуда не сохраняется — это не реализация загрузки.
"""

from __future__ import annotations

from aiogram import Router
from aiogram.types import Message
from redis.asyncio import Redis

from core.logging import get_logger

log = get_logger(__name__)

REPLY = (
    "Пока я умею только здороваться — приём материалов ещё не готов.\n\n"
    "Файл не сохранён, присылать заново не нужно."
)


def _content_kind(message: Message) -> str:
    """Тип присланного — для приоритизации работ по источникам (§19.4).

    Логируется только тип, содержимое не трогаем.
    """
    for attr in ("photo", "document", "voice", "video", "audio", "sticker", "video_note"):
        if getattr(message, attr, None):
            return attr
    if message.text:
        return "text"
    return "other"


def build_fallback_router(redis: Redis, cooldown_sec: int) -> Router:
    """Роутер-перехватчик. Зависимости передаются явно, как в middleware."""
    router = Router(name="fallback")

    @router.message()
    async def handle_unsupported(message: Message) -> None:
        kind = _content_kind(message)
        user_id = message.from_user.id if message.from_user else None
        log.info("unsupported_input", kind=kind, user_id=user_id)

        if user_id is not None and not await _may_reply(redis, user_id, cooldown_sec):
            # Альбом из десяти фотографий приходит десятью сообщениями.
            # Без этого пользователь получил бы десять одинаковых ответов.
            return

        await message.answer(REPLY)

    return router


async def _may_reply(redis: Redis, user_id: int, cooldown_sec: int) -> bool:
    """Правда, если за окно ответа ещё не было.

    Ключ ставится с `NX`, поэтому окно отсчитывается от первого ответа и не
    сдвигается каждым следующим сообщением.
    """
    key = f"unsupported:{user_id}"
    first = await redis.set(key, "1", ex=cooldown_sec, nx=True)
    return bool(first)
