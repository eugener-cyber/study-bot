"""Точка входа бота.

ТЗ §2.2: подключение к локальному Bot API server вместо публичного — лимит
файла поднимается с 20 МБ до 2 ГБ, и файлы доступны по пути на диске.

ТЗ §2: FSM на Redis. Dispatcher без storage молча берёт MemoryStorage, и
состояние терялось бы при рестарте — обнаружилось бы это только на приёмке
WP-05 «после down && up состояние не теряется».
"""

from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.fsm.storage.redis import RedisStorage
from redis.asyncio import Redis

from bot.handlers import start
from bot.middlewares.auth import AuthMiddleware
from bot.middlewares.errors import ErrorsMiddleware
from bot.middlewares.throttle import ThrottleMiddleware
from core.config import Settings, load_settings
from core.logging import configure_logging, get_logger

log = get_logger(__name__)

LOCAL_BOT_API = "http://telegram-bot-api:8081"


def build_dispatcher(settings: Settings, redis: Redis) -> Dispatcher:
    """Собирает диспетчер с middleware и роутерами.

    Порядок регистрации важен: errors -> auth -> throttle. Перехватчик ошибок
    стоит первым, чтобы исключение в любом последующем слое дошло до него.
    """
    dispatcher = Dispatcher(storage=RedisStorage(redis=redis))

    for observer in (dispatcher.message, dispatcher.callback_query):
        observer.middleware(ErrorsMiddleware())
        observer.middleware(AuthMiddleware(settings.ALLOWED_USER_IDS))
        observer.middleware(
            ThrottleMiddleware(
                redis=redis,
                limit=settings.THROTTLE_MESSAGES,
                window_sec=settings.THROTTLE_WINDOW_SEC,
            )
        )

    dispatcher.include_router(start.router)
    return dispatcher


async def run() -> None:
    configure_logging()
    settings = load_settings()

    redis = Redis.from_url(settings.REDIS_URL)
    # Проверяем связь сразу: без Redis не работают ни FSM, ни троттлинг, и
    # падать лучше на старте, чем на первом сообщении пользователя.
    await redis.ping()

    session = AiohttpSession(api=TelegramAPIServer.from_base(LOCAL_BOT_API))
    bot = Bot(token=settings.BOT_TOKEN, session=session)
    dispatcher = build_dispatcher(settings, redis)

    log.info("bot_starting", allowed_users=len(settings.ALLOWED_USER_IDS))
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(run())
