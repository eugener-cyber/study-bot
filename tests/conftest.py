"""Общие фикстуры.

Сеть в тестах запрещена: ТЗ §36 допускает реальные вызовы только под маркером
`live`. Используется pytest-socket, а не подмена socket.connect руками — та не
перехватывает getaddrinfo и ломается при смене внутренностей asyncio.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from pytest_socket import disable_socket, enable_socket

ENV_STUB = {
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


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Подставляет обязательные переменные §30.2 фиктивными значениями.

    Без этого Settings валится на старте — и это правильное поведение, но в
    тестах нам нужно проверять не его, а то, что идёт дальше.
    """
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for key, value in ENV_STUB.items():
        monkeypatch.setenv(key, value)
    # .env рядом с репозиторием не должен влиять на тесты.
    monkeypatch.setenv("BOT_TOKEN", ENV_STUB["BOT_TOKEN"])
    yield
