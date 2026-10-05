"""Окружение Alembic.

URL берётся из `Settings`, а не из `alembic.ini`: §30.2 называет
`DATABASE_URL` обязательной переменной, и второй источник настроек разошёлся
бы с первым молча.

`compare_type` и `compare_server_default` выставлены **явно**. Опираться на
значения по умолчанию нельзя: у `compare_server_default` дефолт `False`, то
есть `alembic check` молча не заметил бы расхождения в default — а в §3
двадцать семь колонок с `DEFAULT`, и у `questions.difficulty` половина
свойства держится именно на нём. Дефолты вдобавок меняются между версиями
Alembic, а эта запись не меняется.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context
from core.config import load_settings
from core.db.engine import build_engine
from core.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )


def run_migrations_offline() -> None:
    """Режим без подключения — печатает SQL."""
    context.configure(
        url=load_settings().DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = build_engine(load_settings().DATABASE_URL)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    """Режим с подключением. Движок берётся из `core.db.engine` — один на проект."""
    connectable = config.attributes.get("connection", None)
    if connectable is not None:
        # Путь для тестов и для `alembic check`: соединение передаётся снаружи.
        _run(connectable)
        return
    asyncio.run(_run_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
