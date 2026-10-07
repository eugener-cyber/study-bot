"""Сборка диспетчера: порядок слоёв и покрытие всех типов апдейтов.

Каждый middleware проверен в изоляции в своём файле. Здесь проверяется то, что
из них собрано, — и до ревью PR #3 этого не проверял никто. Проверяющий показал
мутациями: перевёрнутый порядок middleware и роутер-перехватчик, включённый
раньше `/start`, оставляли прогон зелёным (74 passed в обоих случаях).

Тесты идут через `Dispatcher.feed_update`, то есть через ту же точку, в которую
попадает настоящий апдейт из поллинга. Сеть не нужна: исходящие вызовы
перехватывает `RecordingSession`.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage, TelegramMethod
from aiogram.types import CallbackQuery, Chat, InlineQuery, Message, PhotoSize, Update, User

from bot.errors import SAFE_REPLY
from bot.handlers.fallback import REPLY
from bot.handlers.start import CONSENT_CALLBACK, GREETING
from bot.main import build_dispatcher
from core.config import Settings
from tests.conftest import noop_sessions

ALLOWED_ID = 111
OUTSIDER_ID = 999

TELEGRAM_NOT_MODIFIED = (
    "Bad Request: message is not modified: specified new message content and reply "
    "markup are exactly the same as a current content and reply markup of the message"
)
"""Ответ Telegram при правке сообщения на тот же текст — дословно.

Записан литералом, а не взят из `bot.handlers.start.NOT_MODIFIED`. Иначе фейк и
рабочий код строятся из одной константы, пара самосогласованна при любом её
значении, и расхождение константы с реальностью никакой тест не заметит —
вернулся бы FAIL 2.2: пользователь видит «Что-то пошло не так» на успешно
принятом согласии. Проверяющий показал это, подменив константу: прогон остался
зелёным.
"""


class RecordingSession(BaseSession):
    """Сессия, которая ничего не отправляет и всё записывает."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self.not_modified_collisions = 0
        self._edited: dict[tuple[Any, Any], str] = {}

    async def close(self) -> None:
        return None

    async def make_request(
        self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None
    ) -> Any:
        self.calls.append(method)
        if isinstance(method, EditMessageText):
            # Воспроизводим поведение Telegram: правка на тот же текст — ошибка.
            # Без этого двойное нажатие в тесте проходит, а в чате даёт сбой.
            key = (method.chat_id, method.message_id)
            if self._edited.get(key) == method.text:
                self.not_modified_collisions += 1
                raise TelegramBadRequest(method=method, message=TELEGRAM_NOT_MODIFIED)
            self._edited[key] = method.text or ""
            return True
        if isinstance(method, SendMessage):
            return Message(
                message_id=2,
                date=datetime.datetime(2026, 1, 1),
                chat=Chat(id=1, type="private"),
            )
        return True

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        yield b""

    def texts(self) -> list[str]:
        """Тексты отправленных сообщений."""
        return [c.text for c in self.calls if isinstance(c, SendMessage)]

    def edits(self) -> list[str]:
        """Тексты правок сообщений — отдельно от отправленных.

        Нужны, потому что ответ на нажатие кнопки идёт правкой экрана, а не
        новым сообщением: проверка по `replies()` его не видит.
        """
        return [c.text or "" for c in self.calls if isinstance(c, EditMessageText)]

    def replies(self) -> list[str]:
        """Всё, что пользователь увидит: и сообщения, и алерты на callback.

        Нужна отдельно от `texts`: перехватчик ошибок отвечает на callback
        через `AnswerCallbackQuery`, и проверка только по `SendMessage`
        пропускала безопасное сообщение. Мутационный прогон это показал —
        тест двойного нажатия оставался зелёным без правки обработчика.
        """
        out = [c.text for c in self.calls if isinstance(c, SendMessage)]
        out += [
            c.text for c in self.calls if isinstance(c, AnswerCallbackQuery) and c.text is not None
        ]
        return out


def _settings() -> Settings:
    return Settings(  # type: ignore[arg-type]
        _env_file=None,
        BOT_TOKEN="0000000000:X",
        TELEGRAM_API_ID="1",
        TELEGRAM_API_HASH="h" * 32,
        ALLOWED_USER_IDS=str(ALLOWED_ID),
        DATABASE_URL="postgresql+asyncpg://t:t@localhost/t",
        REDIS_URL="redis://localhost:6379/0",
        TZ_DEFAULT="Europe/Moscow",
        LLM_PROVIDER="manual",
    )


def _redis() -> AsyncMock:
    """Redis, который устраивает и RedisStorage, и троттлинг, и кулдаун."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    pipe = AsyncMock()
    pipe.incr = lambda *a, **k: None
    pipe.expire = lambda *a, **k: None
    pipe.execute = AsyncMock(return_value=[1, True])
    redis.pipeline = lambda *a, **k: pipe
    return redis


def _bot() -> tuple[Bot, RecordingSession]:
    session = RecordingSession()
    return Bot(token="0000000000:TEST-TOKEN-NOT-REAL-0000000000000", session=session), session


def _message_update(user_id: int, **extra: Any) -> Update:
    return Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=datetime.datetime(2026, 1, 1),
            chat=Chat(id=user_id, type="private"),
            from_user=User(id=user_id, is_bot=False, first_name="T"),
            **extra,
        ),
    )


def _inline_update(user_id: int) -> Update:
    return Update(
        update_id=2,
        inline_query=InlineQuery(
            id="q1",
            from_user=User(id=user_id, is_bot=False, first_name="T"),
            query="что-нибудь",
            offset="",
        ),
    )


async def test_start_is_answered_for_allowed_user() -> None:
    bot, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())

    await dispatcher.feed_update(bot, _message_update(ALLOWED_ID, text="/start"))

    assert session.texts() == [GREETING]


async def test_fallback_router_does_not_swallow_start() -> None:
    """Порядок роутеров: перехватчик §19.4 включён последним.

    Проверяющий включил его первым — прогон остался зелёным, хотя `/start`
    перестал работать. Ассерт на точный текст это ловит.
    """
    bot, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())

    await dispatcher.feed_update(bot, _message_update(ALLOWED_ID, text="/start"))

    assert REPLY not in session.texts()


async def test_unsupported_input_reaches_fallback() -> None:
    bot, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())
    photo = [PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)]

    await dispatcher.feed_update(bot, _message_update(ALLOWED_ID, photo=photo))

    assert session.texts() == [REPLY]


async def test_outsider_gets_nothing_at_all() -> None:
    """Единственный критерий приёмки, который проверялся только вручную."""
    bot, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())

    await dispatcher.feed_update(bot, _message_update(OUTSIDER_ID, text="/start"))

    assert session.calls == []


async def test_outsider_is_rejected_before_throttle_counter() -> None:
    """Порядок middleware: auth раньше throttle.

    Считать обращения постороннего не нужно — он отклонён раньше. Если слои
    поменять местами, счётчик в Redis начнёт расти от чужих обращений.
    """
    bot, _ = _bot()
    redis = _redis()
    dispatcher = build_dispatcher(_settings(), redis)
    touched: list[str] = []
    redis.pipeline = lambda *a, **k: touched.append("pipeline")  # type: ignore[assignment]

    await dispatcher.feed_update(bot, _message_update(OUTSIDER_ID, text="привет"))

    assert touched == []


async def test_whitelist_covers_update_types_without_handlers_today() -> None:
    """Белый список действует на любой тип апдейта, а не на два из них.

    До ревью auth стоял на обсерверах `message` и `callback_query`. Здесь
    хендлер для `inline_query` регистрируется уже после сборки — как это и
    произойдёт в следующих пакетах — и проверяется, что он не вызывается для
    постороннего и вызывается для своего.
    """
    bot, _ = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())
    seen: list[int] = []

    probe = Router(name="probe")

    async def handle_probe(query: InlineQuery) -> None:
        seen.append(query.from_user.id)

    probe.inline_query.register(handle_probe)
    dispatcher.include_router(probe)

    await dispatcher.feed_update(bot, _inline_update(OUTSIDER_ID))
    assert seen == [], "посторонний дошёл до хендлера нового типа апдейта"

    await dispatcher.feed_update(bot, _inline_update(ALLOWED_ID))
    assert seen == [ALLOWED_ID], "свой не дошёл до хендлера нового типа апдейта"


@pytest.mark.parametrize(
    "user_id,redis_broken,expected",
    [
        (ALLOWED_ID, False, [GREETING]),
        (ALLOWED_ID, True, [SAFE_REPLY]),
        (OUTSIDER_ID, False, []),
        (OUTSIDER_ID, True, []),
    ],
    ids=["свой-норма", "свой-сбой", "посторонний-норма", "посторонний-сбой"],
)
async def test_who_and_what_broke_are_crossed(
    user_id: int, redis_broken: bool, expected: list[str]
) -> None:
    """Два измерения перекрещены: кто обратился и что сломалось.

    Ревью фазы 2 нашло дефект, внесённый правкой FAIL 2.1: перехватчик ошибок
    на наблюдателе `errors` стоит снаружи цепочки middleware, то есть снаружи
    `AuthMiddleware`, и отвечал постороннему при любом сбое до авторизации.
    Прежние тесты этого не ловили, потому что проверяли измерения по
    отдельности: сбой — только для своего, постороннего — только при здоровом
    Redis. Четвёртая клетка таблицы не проверялась никем.

    Случай «посторонний-сбой» достижим снаружи: Redis перезапускается штатно,
    и посторонний, пишущий боту регулярно, получал в эти секунды
    подтверждение, что адресат живой, — ровно то, что §1.3 запрещает.
    """
    bot, session = _bot()
    redis = _redis()
    if redis_broken:
        redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
    dispatcher = build_dispatcher(_settings(), redis)

    await dispatcher.feed_update(bot, _message_update(user_id, text="/start"))

    assert session.replies() == expected


async def test_fsm_failure_still_answers_user() -> None:
    """Сбой Redis при резолвинге FSM — пользователь получает сообщение (§31).

    Это воспроизведение дефекта FAIL 2.1. Прежний перехватчик был inner
    middleware обсервера и вызывался после `FSMContextMiddleware`, поэтому
    падение Redis давало пользователю тишину. Redis перезапускается штатно,
    и тишина в эти секунды — ровно та картина, из-за которой пришлось вводить
    переходный §19.4.
    """
    bot, session = _bot()
    redis = _redis()
    redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
    dispatcher = build_dispatcher(_settings(), redis)

    await dispatcher.feed_update(bot, _message_update(ALLOWED_ID, text="/start"))

    assert session.replies() == [SAFE_REPLY]


async def test_handler_failure_still_answers_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Падение в самом хендлере — тот же безопасный ответ."""
    import bot.handlers.start as start_module

    async def boom(_message: Message) -> None:
        raise RuntimeError("хендлер сломался")

    monkeypatch.setattr(start_module, "handle_start", boom)

    bot_obj, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis())

    await dispatcher.feed_update(bot_obj, _message_update(ALLOWED_ID, text="/start"))

    assert session.replies() == [SAFE_REPLY]


def _consent_update(user_id: int, update_id: int) -> Update:
    return Update(
        update_id=update_id,
        callback_query=CallbackQuery(
            id=f"cb{update_id}",
            from_user=User(id=user_id, is_bot=False, first_name="T"),
            chat_instance="x",
            data=CONSENT_CALLBACK,
            message=Message(
                message_id=1,
                date=datetime.datetime(2026, 1, 1),
                chat=Chat(id=user_id, type="private"),
            ),
        ),
    )


async def test_double_consent_click_shows_no_failure(no_consent_write: list[int]) -> None:
    """Сквозная проверка FAIL 2.2 на собранном диспетчере.

    `RecordingSession` воспроизводит поведение Telegram: вторая правка на тот
    же текст даёт «message is not modified». До правки это доходило до
    перехватчика ошибок, и пользователь видел «Что-то пошло не так» на
    успешно принятом согласии.
    """
    bot, session = _bot()
    dispatcher = build_dispatcher(_settings(), _redis(), noop_sessions)  # type: ignore[arg-type]

    await dispatcher.feed_update(bot, _consent_update(ALLOWED_ID, 10))
    await dispatcher.feed_update(bot, _consent_update(ALLOWED_ID, 11))

    assert no_consent_write == [ALLOWED_ID, ALLOWED_ID], "согласие не доходило до сервиса"

    # Положительный ассерт: столкновение действительно произошло. Остальные
    # проверки имеют форму «плохого не случилось» и выполняются, когда опасная
    # ситуация просто не возникла — Проверяющий отключил воспроизведение
    # ошибки в фейке и получил зелёный прогон.
    assert session.not_modified_collisions == 1, "столкновения не было, тест пустой"
    assert sum(isinstance(c, EditMessageText) for c in session.calls) == 2
    assert SAFE_REPLY not in session.replies()


async def test_safe_reply_leaks_nothing() -> None:
    assert "Traceback" not in SAFE_REPLY
    assert "Error" not in SAFE_REPLY


def test_dispatcher_can_be_built_twice() -> None:
    """Фабрики роутеров вместо модульных объектов (FAIL 4.3).

    С `router = Router(...)` на уровне модуля второй вызов падал
    `RuntimeError: Router is already attached`, и всё выше было невозможно.
    """
    first = build_dispatcher(_settings(), _redis())
    second = build_dispatcher(_settings(), _redis())
    assert isinstance(first, Dispatcher) and isinstance(second, Dispatcher)
    assert first is not second


def test_answer_callback_query_is_a_known_method() -> None:
    """Страховка от переименования метода в aiogram: `RecordingSession` молча
    вернула бы True, и тесты на callback стали бы бессмысленными."""
    assert issubclass(AnswerCallbackQuery, TelegramMethod)
