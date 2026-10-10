"""Сборка провайдера и клиента по настройкам. ТЗ §2.1.

Отдельный модуль, потому что собирать клиент нужно в трёх местах — воркер,
харнесс `make seed`, будущие точки §11.4 и §21.3 — а знание о том, какой
провайдер отвечает за значение `LLM_PROVIDER`, должно быть одно. Три похожих
сборки разошлись бы в мелочах: один путь получил бы `DatabaseRecorder`, другой
забыл бы его, и часть вызовов исчезла бы из §6.4 молча.

`anthropic` и `yandexgpt` поднимают ошибку с указанием причины, а не
`NotImplementedError` без текста: §2.1 называет три особенности Anthropic,
существенные для архитектуры, и каждая проверяется только живым вызовом.
Непроверенный модуль на критическом пути хуже отсутствующего — отсутствующий
виден.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import Settings
from core.llm.base import LLMProvider
from core.llm.client import DatabaseRecorder, LLMClient
from core.llm.manual import ManualProvider

KNOWN_PROVIDERS = ("manual", "anthropic", "yandexgpt")
"""Значения `LLM_PROVIDER` из §2.1. Реализован один — `manual`."""


def build_provider(settings: Settings) -> LLMProvider:
    """Провайдер по `LLM_PROVIDER` (§2.1, §30.2)."""
    if settings.LLM_PROVIDER == "manual":
        return ManualProvider()

    if settings.LLM_PROVIDER in KNOWN_PROVIDERS:
        raise NotImplementedError(
            f"провайдер {settings.LLM_PROVIDER!r} ещё не реализован: пакет WP-04 "
            "реализует интерфейс §2.1 и провайдер `manual`, потому что ключа "
            "для живого вызова нет и выполнить код провайдера нельзя ни разу. "
            "См. Issue #22 (CR-M) и ADR-0005."
        )

    raise ValueError(
        f"неизвестный LLM_PROVIDER: {settings.LLM_PROVIDER!r}; "
        f"§2.1 называет {', '.join(KNOWN_PROVIDERS)}"
    )


def build_client(settings: Settings, sessions: async_sessionmaker[AsyncSession]) -> LLMClient:
    """Клиент с учётом в `llm_calls` (§6.4).

    Recorder передаётся всегда и аргументом, а не создаётся внутри клиента по
    необязательному параметру: учёт расхода нельзя «забыть включить», и
    отсутствие умолчания — единственное, что это гарантирует.
    """
    return LLMClient(build_provider(settings), settings, DatabaseRecorder(sessions))
