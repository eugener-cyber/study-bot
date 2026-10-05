"""Движок и фабрика сессий.

Единственная точка создания соединений: ТЗ §35 требует, чтобы доступ к БД шёл
через `core/services`, а сессия прокидывалась middleware. Если движок можно
создать где угодно, граница слоёв перестаёт быть проверяемой — достаточно
одного `create_async_engine` в хендлере.

Фабрика, а не объект уровня модуля. Конвенция установлена в WP-01
(`build_start_router`, `build_fallback_router`, `build_error_handler`) после
того, как модульный `Router` сделал сборку диспетчера непроверяемой: второй
вызов в том же процессе падал. Движок уровня модуля дал бы то же — в тестах
нельзя было бы поднять второй с другой базой.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def build_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Создаёт async engine на `DATABASE_URL` (§30.2).

    `pool_pre_ping` включён: Postgres перезапускается штатно, а без проверки
    соединения первый запрос после перезапуска падает на устаревшем из пула —
    ровно та авария, из-за которой в WP-01 потребовалось вести перехват ошибок
    снаружи FSM.
    """
    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Фабрика сессий.

    `expire_on_commit=False`: после коммита сервис возвращает объект хендлеру,
    и при истечении атрибутов тот обратился бы к БД сам — то есть нарушил бы
    §35, причём незаметно, через ленивую подгрузку.
    """
    return async_sessionmaker(engine, expire_on_commit=False)
