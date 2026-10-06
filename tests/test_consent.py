"""Фиксация согласия на обработку данных (§31, CR-B).

Перенос из WP-01: там нажатие обрабатывалось, но в базу не писалось, потому
что таблицы `users` не существовало. Здесь перенос закрывается.

Тесты идут и через сервис, и через собранный диспетчер. Второе существенно:
хендлер получает сессию из middleware, и проверка только сервиса оставила бы
непокрытым путь, которым пользуется приложение, — ровно дыра `NoDecode` из
WP-01, где все 58 тестов шли через аргументы конструктора.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import CallbackQuery, Chat, Message, Update
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bot.handlers.start import CONSENT_ACCEPTED, CONSENT_CALLBACK, GREETING
from bot.main import build_dispatcher
from core.db.models import User
from core.db.session import session_scope
from core.services.users import accept_consent, find_by_tg_id
from tests.test_dispatcher import ALLOWED_ID, RecordingSession, _redis, _settings

pytestmark = pytest.mark.db


def _bot() -> tuple[Bot, RecordingSession]:
    session = RecordingSession()
    return Bot(token="0000000000:TEST-TOKEN-NOT-REAL-0000000000000", session=session), session


def _start_update(user_id: int, update_id: int = 1) -> Update:
    return Update(
        update_id=update_id,
        message=Message(
            message_id=1,
            date=datetime.datetime(2026, 1, 1),
            chat=Chat(id=user_id, type="private"),
            from_user=TgUser(id=user_id, is_bot=False, first_name="T"),
            text="/start",
        ),
    )


def _consent_update(user_id: int, update_id: int = 2) -> Update:
    return Update(
        update_id=update_id,
        callback_query=CallbackQuery(
            id=f"cb{update_id}",
            from_user=TgUser(id=user_id, is_bot=False, first_name="T"),
            chat_instance="x",
            data=CONSENT_CALLBACK,
            message=Message(
                message_id=1,
                date=datetime.datetime(2026, 1, 1),
                chat=Chat(id=user_id, type="private"),
            ),
        ),
    )


async def _consent_at(engine: AsyncEngine, tg_id: int) -> datetime.datetime | None:
    async with engine.connect() as connection:
        return (
            await connection.execute(select(User.consent_at).where(User.tg_id == tg_id))
        ).scalar_one_or_none()


# --- Сервис ----------------------------------------------------------------


async def test_consent_creates_user_with_timestamp(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    async with session_scope(sessions) as session:
        user = await accept_consent(session, ALLOWED_ID)
        assert user.tg_id == ALLOWED_ID

    assert await _consent_at(engine, ALLOWED_ID) is not None


async def test_second_consent_does_not_overwrite_the_first(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    """`consent_at` — запись о том, когда человек согласился.

    Сдвигать её вперёд при повторном нажатии неверно: юридически значим
    первый момент. За это отвечает `COALESCE` в `ON CONFLICT`.
    """
    async with session_scope(sessions) as session:
        await accept_consent(session, ALLOWED_ID)
    first = await _consent_at(engine, ALLOWED_ID)

    async with session_scope(sessions) as session:
        await accept_consent(session, ALLOWED_ID)
    second = await _consent_at(engine, ALLOWED_ID)

    assert first is not None
    assert first == second


async def test_consent_is_idempotent_by_row_count(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    """Трижды нажатая кнопка не даёт трёх пользователей.

    Один оператор `ON CONFLICT` вместо «прочитать, проверить, вставить»:
    проверка в коде не закрывает гонку двух одновременных нажатий, закрывает
    уникальный индекс `users.tg_id`.
    """
    for _ in range(3):
        async with session_scope(sessions) as session:
            await accept_consent(session, ALLOWED_ID)

    async with engine.connect() as connection:
        count = (await connection.execute(select(User.id).where(User.tg_id == ALLOWED_ID))).all()
    assert len(count) == 1


async def test_find_by_tg_id_returns_none_for_unknown(
    sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    async with session_scope(sessions) as session:
        assert await find_by_tg_id(session, 123_456) is None


# --- Через собранный диспетчер ---------------------------------------------


async def test_consent_click_writes_to_database(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    """Путь приложения: нажатие кнопки доходит до базы.

    Это и есть закрытие критерия WP-01 «`/start` с фиксацией согласия», и
    закрытие замечания FAIL 4.5 из ревью WP-01: там имя теста обещало «без
    обращения к БД», а проверить было нечего, потому что базы не было.
    """
    bot, outgoing = _bot()
    dispatcher = build_dispatcher(_settings(), _redis(), sessions)

    await dispatcher.feed_update(bot, _consent_update(ALLOWED_ID))

    assert await _consent_at(engine, ALLOWED_ID) is not None
    # Ассерт по правкам, а не по отправленным сообщениям: ответ на нажатие
    # идёт `EditMessageText`. Первая версия этой проверки заканчивалась
    # `or outgoing.calls` и потому была истинной почти всегда.
    assert outgoing.edits() == [CONSENT_ACCEPTED]


async def test_start_does_not_create_user_before_consent(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    """`/start` строку не создаёт — согласия ещё нет.

    Telegram id — персональные данные (§30.2), и их хранение и есть то, на
    что человек даёт согласие. Создавать запись до нажатия кнопки значило бы
    начать обработку раньше разрешения.
    """
    bot, outgoing = _bot()
    dispatcher = build_dispatcher(_settings(), _redis(), sessions)

    await dispatcher.feed_update(bot, _start_update(ALLOWED_ID))

    assert outgoing.texts() == [GREETING]
    async with engine.connect() as connection:
        rows = (await connection.execute(select(User.id))).all()
    assert rows == [], "строка создана до согласия"


async def test_failed_write_does_not_report_success(
    sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine, clean_tables: None
) -> None:
    """Сбой записи даёт безопасное сообщение, а не «Готово, согласие принято».

    Порядок в хендлере выбран так намеренно: запись идёт до ответа. Обратный
    порядок давал бы ответ об успехе при пустом `consent_at` — и человек
    считал бы согласие данным, а в базе его не было бы.
    """
    bot, outgoing = _bot()

    class BrokenSession:
        """Сессия, падающая на первом же запросе.

        Полноценный фейк, а не `AsyncMock`: мок отдавал бы корутину, которую
        никто не ждёт, и тест проходил бы на `TypeError` в `async with` —
        то есть по другой причине, чем задумано. Предупреждение
        «coroutine was never awaited» это и показало.
        """

        async def __aenter__(self) -> BrokenSession:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def execute(self, *_: object, **__: object) -> object:
            raise RuntimeError("база недоступна")

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    def broken_sessions() -> BrokenSession:
        return BrokenSession()

    dispatcher = build_dispatcher(_settings(), _redis(), broken_sessions)  # type: ignore[arg-type]

    await dispatcher.feed_update(bot, _consent_update(ALLOWED_ID))

    # И правки, и отправленные сообщения: ответ на нажатие идёт правкой, и
    # проверка только по `replies()` мутацию «ответ раньше записи» пропускала.
    assert CONSENT_ACCEPTED not in outgoing.edits()
    assert CONSENT_ACCEPTED not in outgoing.replies()
    assert await _consent_at(engine, ALLOWED_ID) is None


async def test_handler_does_not_touch_session_directly(
    sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Хендлер не обращается к БД сам — только через сервис (§35).

    Проверяется по объекту сессии, а не по факту вызова сервиса: вызов сервиса
    прошёл бы и у хендлера, который вызвал сервис и вдобавок полез в базу.
    Замечание ревью плана WP-02.
    """
    from bot.handlers.start import handle_consent

    calls: list[str] = []

    class Spy:
        def __init__(self, inner: AsyncSession) -> None:
            self._inner = inner

        async def execute(self, *args: object, **kwargs: object) -> object:
            calls.append("execute")
            return await self._inner.execute(*args, **kwargs)  # type: ignore[arg-type]

        def add(self, *args: object, **kwargs: object) -> None:
            calls.append("add")

        async def commit(self) -> None:
            calls.append("commit")

    async with session_scope(sessions) as real:
        spy = Spy(real)
        callback = _consent_update(ALLOWED_ID).callback_query
        assert callback is not None
        object.__setattr__(callback, "answer", AsyncMock())
        if isinstance(callback.message, Message):
            object.__setattr__(callback.message, "edit_text", AsyncMock())

        await handle_consent(callback, spy)  # type: ignore[arg-type]

    # `execute` вызывает сервис — это разрешено. `add` и `commit` из хендлера
    # означали бы работу с БД в обход сервисного слоя.
    assert "add" not in calls
    assert "commit" not in calls
    assert calls.count("execute") > 0, "сервис до базы не дошёл"
