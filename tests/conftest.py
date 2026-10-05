"""Общие фикстуры.

Сеть в тестах запрещена: ТЗ §36 допускает реальные вызовы только под маркером
`live`, а §41.4 требует, чтобы такие тесты не участвовали в CI. Здесь живёт
первая половина — блокировка сокетов; вторая половина — фильтр `-m "not live"`
в `addopts` (`pyproject.toml`), и её держит канарейка в `test_checks_config.py`.

Используется pytest-socket, а не подмена socket.connect руками — та не
перехватывает getaddrinfo и ломается при смене внутренностей asyncio.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pytest_socket import disable_socket, enable_socket

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


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Блокирует сеть везде, кроме тестов с маркером `live`."""
    if request.node.get_closest_marker("live"):
        enable_socket()
        yield
        return
    # allow_unix_socket: asyncio строит self-pipe через socketpair(AF_UNIX).
    # Без этого блокируется само создание событийного цикла, а не сеть.
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()
