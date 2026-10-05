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

from bot.errors import handle_error
from bot.handlers import fallback, start
from bot.middlewares.auth import AuthMiddleware
from bot.middlewares.throttle import ThrottleMiddleware
from core.config import Settings, load_settings
from core.logging import configure_logging, get_logger

log = get_logger(__name__)

LOCAL_BOT_API = "http://telegram-bot-api:8081"


def build_dispatcher(settings: Settings, redis: Redis) -> Dispatcher:
    """Собирает диспетчер: перехватчик ошибок, авторизация, троттлинг, роутеры.

    Перехват ошибок — обработчик наблюдателя `errors`, а не middleware. Свой
    middleware, даже outer на `update`, встаёт в очередь **после** собственных
    middleware диспетчера (`UserContextMiddleware`, `FSMContextMiddleware`),
    потому что те регистрируются в `Dispatcher.__init__`. Сбой Redis при
    резолвинге FSM до него не доходил, и пользователь получал тишину вместо
    безопасного сообщения — ревью PR #3, FAIL 2.1. Наблюдатель `errors`
    вызывается самым внешним middleware самого aiogram и покрывает всё: FSM,
    фильтры, хендлеры и middleware ниже.

    Авторизация и троттлинг — outer middleware на `update`, а не на обсерверах
    `message` и `callback_query`. На обсерверах они покрывали два типа апдейтов
    из примерно четырнадцати, и первый же `inline_query` или `my_chat_member`
    в следующих пакетах попал бы в хендлер без белого списка (§1.3) — FAIL 2.3.
    `event_from_user` на этом уровне уже заполнен: `UserContextMiddleware`
    стоит в очереди раньше.

    Порядок внутри: auth -> throttle. Считать обращения постороннего не нужно,
    он отклонён раньше.
    """
    dispatcher = Dispatcher(storage=RedisStorage(redis=redis))

    dispatcher.errors.register(handle_error)

    dispatcher.update.outer_middleware(AuthMiddleware(settings.ALLOWED_USER_IDS))
    dispatcher.update.outer_middleware(
        ThrottleMiddleware(
            redis=redis,
            limit=settings.THROTTLE_MESSAGES,
            window_sec=settings.THROTTLE_WINDOW_SEC,
        )
    )

    dispatcher.include_router(start.build_start_router())
    # Последним: ловит всё, что не разобрали хендлеры выше (§19.4, переходное).
    dispatcher.include_router(
        fallback.build_fallback_router(redis, settings.UNSUPPORTED_REPLY_COOLDOWN_SEC)
    )
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
