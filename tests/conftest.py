"""Общие фикстуры.

Сеть в тестах запрещена: ТЗ §36 допускает реальные вызовы только под маркером
`live`, а §41.4 требует, чтобы такие тесты не участвовали в CI. Здесь живёт
первая половина — блокировка сокетов; вторая половина — фильтр `-m "not live"`
в `addopts` (`pyproject.toml`), и её держит канарейка в `test_checks_config.py`.

Используется pytest-socket, а не подмена socket.connect руками — та не
перехватывает getaddrinfo и ломается при смене внутренностей asyncio.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import asyncpg
import pytest
import pytest_asyncio
from alembic.config import Config
from pytest_socket import disable_socket, enable_socket, socket_allow_hosts
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from alembic import command
from core.config import load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.models import Base

ENV_STUB = {
    # Обязательные переменные §30.2 фиктивными значениями: без них Settings
    # валится на старте. Используется `_from_env` в `test_config.py` — путь
    # через настоящее окружение, а не через аргументы конструктора.
    "BOT_TOKEN": "0000000000:TEST-TOKEN-NOT-REAL-0000000000000",
    "TELEGRAM_API_ID": "1",
    "TELEGRAM_API_HASH": "00000000000000000000000000000000",
    "ALLOWED_USER_IDS": "111,222",
    "DATABASE_URL": "postgresql+asyncpg://t:t@localhost:5432/t",
    "REDIS_URL": "redis://localhost:6379/0",
    "TZ_DEFAULT": "Europe/Moscow",
    "LLM_PROVIDER": "manual",
}


DB_HOST = "127.0.0.1"
"""Единственный адрес, к которому тестам разрешено подключаться.

Решение Архитектора по вопросу 3 плана WP-02: «при отключённой сети» в §41.4
читается как запрет на внешнюю сеть — её смысл в том, чтобы тесты не ходили к
LLM-провайдеру. Postgres доступен на loopback и внешней сетью не является.
Разрешение точечное, а не снятие блокировки: тест без маркера `db` в сеть
по-прежнему не выйдет, включая выход на loopback.
"""

TEST_DB_NAME = "studybot_test"
"""Отдельная база в том же контейнере: `make reset` не должен сносить данные
разработчика, а тесты — зависеть от того, что в них лежит."""


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Блокирует сеть. Исключения: маркер `live` и маркер `db`.

    `live` — тесты, которым нужна настоящая сеть (§36); в CI они не участвуют.
    `db` — тесты схемы, которым нужен Postgres на loopback.
    """
    if request.node.get_closest_marker("live"):
        enable_socket()
        yield
        return

    if request.node.get_closest_marker("db"):
        # Порядок важен: `socket_allow_hosts` подменяет `connect`, а
        # `disable_socket` подменяет сам класс `socket.socket`. Вызванный после
        # `disable_socket`, он патчил бы уже заблокированный класс, и
        # подключение падало бы на создании сокета, а не на адресе.
        enable_socket()
        socket_allow_hosts([DB_HOST], allow_unix_socket=True)
        yield
        enable_socket()
        return

    # allow_unix_socket: asyncio строит self-pipe через socketpair(AF_UNIX).
    # Без этого блокируется само создание событийного цикла, а не сеть.
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()


def _admin_url() -> str:
    """URL к служебной базе: создавать и дропать тестовую нужно не из неё самой."""
    return load_settings().DATABASE_URL


def _test_url() -> str:
    admin = _admin_url()
    return admin.rsplit("/", 1)[0] + "/" + TEST_DB_NAME


def alembic_config(url: str) -> Config:
    """Конфигурация Alembic, нацеленная на заданную базу."""
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    return config


async def run_alembic(url: str, action: str, revision: str = "head") -> None:
    """Выполняет команду Alembic на переданной базе.

    Соединение создаётся здесь и передаётся через `config.attributes`, а
    `env.py` его подхватывает: иначе Alembic открыл бы собственное, и накат
    шёл бы в другой транзакции, чем проверка.
    """
    config = alembic_config(url)
    engine = build_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_conn: _invoke(config, sync_conn, action, revision)
            )
    finally:
        await engine.dispose()


def _invoke(config: Config, connection: Connection, action: str, revision: str) -> None:
    config.attributes["connection"] = connection
    if action == "upgrade":
        command.upgrade(config, revision)
    elif action == "downgrade":
        command.downgrade(config, revision)
    else:  # pragma: no cover — защита от опечатки в тесте
        raise ValueError(f"неизвестное действие Alembic: {action}")


async def _recreate_test_database() -> None:
    """Пересоздаёт тестовую базу и накатывает миграции.

    Схема поднимается **миграциями**, а не `create_all`: иначе модели и
    миграция разойдутся, тесты на моделях останутся зелёными, а `make reset`
    даст другую схему. Правило записано в `CLAUDE.md` и проверяется тестами
    `test_migrations.py`.
    """
    admin = _admin_url().replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(admin)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await conn.close()

    await run_alembic(_test_url(), "upgrade")


@pytest.fixture(scope="session")
def _database() -> Iterator[None]:
    """Тестовая база на всю сессию прогона."""
    asyncio.run(_recreate_test_database())
    yield


@pytest_asyncio.fixture
async def engine(_database: None) -> AsyncIterator[AsyncEngine]:
    """Движок к тестовой базе."""
    created = build_engine(_test_url())
    try:
        yield created
    finally:
        await created.dispose()


@pytest_asyncio.fixture
async def sessions(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Фабрика сессий к тестовой базе."""
    yield build_sessionmaker(engine)


@pytest_asyncio.fixture
async def clean_tables(engine: AsyncEngine) -> AsyncIterator[None]:
    """Опустошает таблицы перед тестом, не пересоздавая схему.

    `TRUNCATE ... CASCADE` вместо пересоздания базы: схема одна на прогон,
    и её пересборка на каждый тест заняла бы больше, чем сам прогон.
    """
    names = ", ".join(f'"{name}"' for name in Base.metadata.tables)
    async with engine.begin() as connection:
        await connection.execute(sql_text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
    yield
