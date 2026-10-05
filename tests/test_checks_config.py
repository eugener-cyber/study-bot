"""Конфигурация проверок — это требование ТЗ, а не вкусовая настройка.

§36 вводит маркер `live` для тестов, которым нужна сеть. §41.4 требует, чтобы
такие тесты в CI не участвовали. Требование целиком живёт в конфигурационном
файле, поэтому и проверяется здесь: без этого первый же `live`-тест в WP-04
пойдёт в CI и упадёт на pytest-socket — и выглядеть это будет сломанным
тестом, а не сломанной конфигурацией.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _addopts() -> list[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    raw = config["tool"]["pytest"]["ini_options"].get("addopts", [])
    return raw.split() if isinstance(raw, str) else list(raw)


def test_live_tests_excluded_from_default_run() -> None:
    """В прогоне по умолчанию (а значит и в CI) `live`-тесты отфильтрованы."""
    addopts = _addopts()
    assert "-m" in addopts, f"в addopts нет фильтра по маркерам: {addopts}"
    assert "not live" in addopts, f"в addopts нет 'not live': {addopts}"


def test_live_marker_is_declared() -> None:
    """Маркер объявлен — иначе `--strict-markers` в будущем уронит прогон."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("live:") for m in markers), markers


@pytest.mark.live
def test_live_marker_canary() -> None:
    """Канарейка: этот тест обязан быть отброшен, а не выполнен.

    Проверка чтением конфига выше показывает, что фильтр записан. Эта
    показывает, что он ещё и действует: если `-m "not live"` потеряется,
    тест выполнится и прогон станет красным с понятной причиной.
    """
    pytest.fail('live-тест выполнился в прогоне по умолчанию — потерян фильтр -m "not live"')


@pytest.mark.db
def test_db_marker_allows_only_loopback() -> None:
    """Маркер `db` открывает ровно один адрес, а не сеть целиком.

    Решение Архитектора по вопросу 3 плана WP-02: «при отключённой сети» —
    запрет на внешнюю сеть, её смысл в том, чтобы тесты не ходили к
    LLM-провайдеру. Проверяется, что смысл сохранён: loopback доступен,
    посторонний адрес — нет.
    """
    import socket

    with socket.socket() as probe:
        probe.settimeout(0.2)
        probe.connect(("127.0.0.1", 5432))

    with socket.socket() as probe, pytest.raises(SocketConnectBlockedError):
        probe.settimeout(0.2)
        probe.connect(("93.184.216.34", 80))


def test_socket_is_blocked_without_db_marker() -> None:
    """Без маркера `db` закрыт и loopback тоже.

    Это и есть «точечное разрешение, а не снятие блокировки»: если бы
    разрешение было глобальным, этот тест прошёл бы, и первый случайный выход
    в сеть из юнит-теста остался бы незамеченным.
    """
    import socket

    with pytest.raises((SocketBlockedError, SocketConnectBlockedError)), socket.socket() as probe:
        probe.settimeout(0.2)
        probe.connect(("127.0.0.1", 5432))
