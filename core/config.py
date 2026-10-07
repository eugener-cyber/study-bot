"""Конфигурация приложения.

Единственный источник настроек. Обязательные переменные ТЗ §30.2 объявлены без
значений по умолчанию: их отсутствие валит приложение на старте с именем
переменной, а не проявляется сбоем в середине работы.

Пороги ТЗ §30.3 живут здесь же со значениями из таблицы. Литералы в коде
запрещены преамбулой §30.3 — крутить эти числа придётся после первых материалов,
и искать их по исходникам не должно быть нужно.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки, собранные из окружения и файла `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ §30.2
    # Обязательные. Без значений по умолчанию — намеренно.

    BOT_TOKEN: str
    """Токен бота от @BotFather."""

    TELEGRAM_API_ID: int
    """api_id приложения с my.telegram.org. Нужен контейнеру telegram-bot-api (§2.2)."""

    TELEGRAM_API_HASH: str
    """api_hash оттуда же."""

    ALLOWED_USER_IDS: Annotated[list[int], NoDecode]
    """Белый список (§1.3). Реальные Telegram id — персональные данные (§30.2).

    NoDecode обязателен: без него pydantic-settings пытается разобрать значение
    переменной окружения как JSON ещё до валидаторов, и «111,222» падает
    с SettingsError. Валидатор ниже разбирает строку сам.
    """

    DATABASE_URL: str
    REDIS_URL: str
    TZ_DEFAULT: str

    LLM_PROVIDER: str
    """anthropic | yandexgpt | manual (§2.1)."""

    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_VISION_MODEL: str = ""
    COST_CURRENCY: str = "USD"

    WHISPER_MODEL: str = ""
    WHISPER_DEVICE: str = "cpu"

    # ------------------------------------------------------------------ §30.3
    # Настраиваемые пороги. Значения — из таблицы ТЗ.

    MIN_FACT_CONFIDENCE: float = 0.55
    DERIVED_WARN_RATIO: float = 0.40
    FACT_DEDUP_THRESHOLD: int = 90
    MAX_SECTION_TOKENS: int = 60000
    LENGTH_LEAK_FACTOR: float = 1.4
    LEXICAL_LEAK_MARGIN: int = 2
    MAX_REGEN_ATTEMPTS: int = 3
    FUZZY_ACCEPT_THRESHOLD: int = 92
    FAST_MS: int = 8000
    SLOW_MS: int = 45000
    MASTERY_HORIZON_DAYS: int = 30
    WEAK_STRENGTH: float = 0.4
    WEAK_MIN_COVERAGE: float = 0.3
    OVERDUE_DAYS: int = 3
    MIN_SESSION_ITEMS: int = 3
    MIN_PUSH_INTERVAL_MIN: int = 90
    ABANDON_AFTER_HOURS: int = 6
    LLM_CONCURRENCY: int = 16
    THROTTLE_MESSAGES: int = 20
    THROTTLE_WINDOW_SEC: int = 60
    MAX_SCENES: int = 200
    MIN_IMAGE_AREA_RATIO: float = 0.02
    NEW_FACT_DELAY_MIN: int = 10
    AWAITING_USER_TIMEOUT_H: int = 24
    MAX_INTERVAL_DAYS: int = 180
    LANG_CONFIDENCE_MIN: float = 0.8

    RECOGNITION_ONLY_CAP: float = 0.7
    """§14.1, CR-1. Потолок `score_i` для факта, по которому нет ни одного
    верного ответа на порождающее задание (`open`).

    Узнавание среди вариантов и воспроизведение по памяти — разные операции с
    разной прочностью следа. Без этого потолка mastery показывала освоение,
    которого нет: «освоено 85%» могло означать «узнаю среди четырёх».
    Применяется в WP-16."""

    GIVEUP_DELAY_SEC: int = 20
    """§20, CR-4. Задержка кнопок «Подсказка» и «Не знаю» на открытом вопросе.

    `0` возвращает прежнее поведение — кнопки сразу. Правка трогает ощущение
    от продукта, поэтому выключается конфигурацией, а не правкой кода.
    Применяется в WP-14."""
    COST_REESTIMATE_FACTOR: float = 1.5

    COST_WARN_THRESHOLD: Decimal | None = Field(default=None)
    """TODO(owner) в §30.3 — значение выводится из спайка и требуется с WP-03.

    Тип Decimal, а не int: §6.3 сравнивает порог с денежной оценкой, а в §3
    стоимость хранится как NUMERIC. Decimal с int Python сравнил бы молча.
    """

    # ------------------------------------------------------------------ разбор

    @field_validator("ALLOWED_USER_IDS", mode="before")
    @classmethod
    def _parse_user_ids(cls, value: object) -> object:
        """Разбирает `123, 456` в список чисел.

        Пустой список отвергается: «открыт всем» недопустимо (§1.3). Поведение
        совпадает с отсутствием переменной — падение на старте с её именем.
        """
        if isinstance(value, str):
            items = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
            if not items:
                raise ValueError(
                    "ALLOWED_USER_IDS пуст. Белый список обязателен: бот без него "
                    "отвечал бы любому, кто найдёт его в поиске Telegram."
                )
            return [int(item) for item in items]
        return value

    @field_validator("ALLOWED_USER_IDS")
    @classmethod
    def _reject_empty_list(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("ALLOWED_USER_IDS пуст — см. §1.3.")
        return value


def load_settings() -> Settings:
    """Собирает настройки. Отдельная функция, чтобы тесты подменяли окружение."""
    # Подавление `call-arg` ниже обязательно и неустранимо: обязательные поля
    # §30.2 приходят из окружения, а mypy видит только сигнатуру `__init__` и
    # требует передать их аргументами. Подавляется ровно один код ошибки, так
    # что опечатка в имени поля или лишний аргумент по-прежнему будут пойманы.
    return Settings()  # type: ignore[call-arg]
