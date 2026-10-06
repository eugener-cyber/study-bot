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
from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.errors import build_error_handler
from bot.handlers import gate, start, upload
from bot.middlewares.auth import AuthMiddleware
from bot.middlewares.db_session import DbSessionMiddleware
from bot.middlewares.throttle import ThrottleMiddleware
from core.config import Settings, load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.logging import configure_logging, get_logger

log = get_logger(__name__)

LOCAL_BOT_API = "http://telegram-bot-api:8081"


def build_dispatcher(
    settings: Settings,
    redis: Redis,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    arq: ArqRedis | None = None,
) -> Dispatcher:
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

    Порядок внутри: auth -> throttle -> сессия БД. Считать обращения
    постороннего не нужно, он отклонён раньше; открывать для него транзакцию и
    занимать соединение из пула — тем более.

    `sessions` необязателен: без него middleware сессии не регистрируется, и
    хендлеры, объявляющие параметр `session`, упадут. Это сделано для тестов,
    которым база не нужна, — а не для работы: `run()` фабрику передаёт всегда.
    """
    dispatcher = Dispatcher(storage=RedisStorage(redis=redis))

    # Белый список передаётся и сюда: наблюдатель `errors` стоит снаружи
    # цепочки middleware, то есть снаружи AuthMiddleware. Без этого бот
    # отвечал бы постороннему при любом сбое до авторизации (§1.3).
    dispatcher.errors.register(build_error_handler(settings.ALLOWED_USER_IDS))

    dispatcher.update.outer_middleware(AuthMiddleware(settings.ALLOWED_USER_IDS))
    dispatcher.update.outer_middleware(
        ThrottleMiddleware(
            redis=redis,
            limit=settings.THROTTLE_MESSAGES,
            window_sec=settings.THROTTLE_WINDOW_SEC,
        )
    )

    if sessions is not None:
        dispatcher.update.outer_middleware(DbSessionMiddleware(sessions))

    dispatcher.include_router(start.build_start_router())
    dispatcher.include_router(gate.build_gate_router(arq))
    dispatcher.include_router(upload.build_upload_router(arq))
    return dispatcher


async def run() -> None:
    """Поднимает бота и ведёт поллинг до остановки процесса.

    Ничего не возвращает и выходит только при остановке поллинга. Соединения
    закрываются в `finally`, включая аварийный выход: иначе при рестарте
    остаётся висящий `getUpdates`, и Telegram отвечает новому процессу
    конфликтом.
    """
    configure_logging()
    settings = load_settings()

    engine = build_engine(settings.DATABASE_URL)
    sessions = build_sessionmaker(engine)

    # Очередь обработки материалов. Отдельное подключение от FSM-хранилища:
    # ARQ держит собственный пул и собственные ключи, и делить объект между
    # двумя библиотеками значит зависеть от того, что ни одна не закроет его
    # раньше другой.
    arq = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))

    redis = Redis.from_url(settings.REDIS_URL)
    # Проверяем связь сразу: без Redis не работают ни FSM, ни троттлинг, и
    # падать лучше на старте, чем на первом сообщении пользователя.
    await redis.ping()

    session = AiohttpSession(api=TelegramAPIServer.from_base(LOCAL_BOT_API))
    bot = Bot(token=settings.BOT_TOKEN, session=session)
    dispatcher = build_dispatcher(settings, redis, sessions, arq)

    log.info("bot_starting", allowed_users=len(settings.ALLOWED_USER_IDS))
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()
        await redis.aclose()
        await arq.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
