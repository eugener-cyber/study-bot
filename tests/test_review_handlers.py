"""Сеанс повторения в чате: `/review` и нажатие варианта. ТЗ §19.1, §20, §22.

Хендлеры вызываются напрямую с поддельными `Message` и `CallbackQuery` — так
же, как тесты кнопок гейта в WP-03. Диспетчер здесь не поднимается: его
сборка и порядок middleware проверяются в `test_dispatcher.py`, а тут —
поведение самих хендлеров.
"""

from __future__ import annotations

import datetime

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bot.callbacks import IDK, Ans
from bot.handlers.review import (
    ALREADY_ANSWERED,
    BROKEN_QUESTION,
    NOTHING_TO_REVIEW,
    handle_answer,
    handle_review,
)
from core.config import Settings
from core.db.models import Question, ReviewLog, Session, SessionItem
from core.services.sessions import current_item
from tests.conftest import ENV_STUB
from tests.learning_factory import (
    MCQ_ANSWER,
    MCQ_PAYLOAD,
    Scaffold,
    add_fact,
    add_question,
    add_review_state,
    scaffold,
)

pytestmark = pytest.mark.db

OWNER_TG = 2100
STRANGER_TG = 2101
CALLBACK_DATA_LIMIT = 64
"""Предел Telegram на `callback_data`, байты. Литерал намеренно: приёмка
WP-14 требует этого от всех фабрик, а первая появляется здесь."""


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, **ENV_STUB)  # type: ignore[arg-type]


class Sent:
    """Журнал отправленных сообщений и порядка вызовов.

    Порядок нужен для §19.1: `answer_callback_query` обязан идти **до** правки
    сообщения. Один общий список, а не два мока по отдельности, — иначе
    порядок между ними не проверить.
    """

    def __init__(self) -> None:
        self.order: list[str] = []
        self.texts: list[str] = []
        self.keyboards: list[object] = []
        self.edits: list[str] = []
        self.alerts: list[str] = []


def _message(sent: Sent, tg_id: int = OWNER_TG) -> Message:
    message = Message(
        message_id=10,
        date=datetime.datetime(2026, 10, 13),
        chat=Chat(id=tg_id, type="private"),
        from_user=TgUser(id=tg_id, is_bot=False, first_name="T"),
        text="/review",
    )

    async def answer(text: str, **kwargs: object) -> Message:
        sent.order.append("send")
        sent.texts.append(text)
        sent.keyboards.append(kwargs.get("reply_markup"))
        return Message(
            message_id=11 + len(sent.texts),
            date=datetime.datetime(2026, 10, 13),
            chat=Chat(id=tg_id, type="private"),
        )

    object.__setattr__(message, "answer", answer)
    return message


def _callback(sent: Sent, data: str, tg_id: int = OWNER_TG) -> CallbackQuery:
    inner = _message(sent, tg_id)

    async def edit_text(text: str, **kwargs: object) -> Message:
        sent.order.append("edit")
        sent.edits.append(text)
        return inner

    object.__setattr__(inner, "edit_text", edit_text)

    callback = CallbackQuery(
        id="cb-1",
        from_user=TgUser(id=tg_id, is_bot=False, first_name="T"),
        chat_instance="x",
        data=data,
        message=inner,
    )

    async def answer(text: str | None = None, **kwargs: object) -> bool:
        sent.order.append("ack")
        if text:
            sent.alerts.append(text)
        return True

    object.__setattr__(callback, "answer", answer)
    return callback


async def _due_facts(engine: AsyncEngine, count: int = 3, *, tg_id: int = OWNER_TG) -> Scaffold:
    base = await scaffold(engine, tg_id=tg_id)
    due_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    for index in range(count):
        fact_id = await add_fact(engine, base, statement=f"Факт {index}")
        await add_question(engine, fact_id)
        await add_review_state(engine, base.user_id, fact_id, due_at=due_at)
    return base


async def test_review_without_candidates_answers_instead_of_silence(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§16: сеанс не собрался — это штатный исход, но не повод молчать.

    Молчание в ответ на команду — ровно тот дефект, из-за которого в WP-03
    существовал переходный §19.4.
    """
    await _due_facts(engine, count=1)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    assert sent.texts == [NOTHING_TO_REVIEW]


async def test_review_sends_task_with_numbered_options_and_keyboard(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§20: вопрос и пронумерованные варианты в тексте, номера в клавиатуре."""
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    assert len(sent.texts) == 1
    text = sent.texts[0]
    assert "Задание 1 из 3" in text
    assert MCQ_PAYLOAD["question"] in text
    assert "1. мел" in text

    keyboard = sent.keyboards[0]
    assert keyboard is not None
    rows = keyboard.inline_keyboard  # type: ignore[union-attr]
    assert [button.text for button in rows[0]] == ["1", "2", "3"]
    assert rows[1][0].text.endswith("Не знаю")


async def test_control_message_id_is_remembered(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§20 и §3: сообщение с клавиатурой запоминается в `sessions`.

    По завершении задания клавиатуру нужно убрать, а убрать её можно только у
    известного сообщения.
    """
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    async with sessions() as session:
        control = (await session.execute(select(Session.control_message_id))).scalar_one()
    assert control is not None


async def test_pressing_an_option_acknowledges_before_editing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§19.1: `answer_callback_query` вызывается до правки сообщения.

    Порядок проверяется по общему журналу вызовов: при медленной правке
    Telegram показывает часики на кнопке, а при ошибке правки нажатие
    осталось бы неподтверждённым вовсе.
    """
    base = await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None

    async with sessions() as session:
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=0).pack()), session, settings)
        await session.commit()

    assert sent.order.count("ack") == 1, "нажатие подтверждено не один раз"
    assert "edit" in sent.order
    assert sent.order.index("ack") < sent.order.index("edit")
    assert "✅ Верно" in sent.edits[0]
    assert "Почему:" in sent.edits[0]
    assert base.user_id is not None


async def test_next_task_arrives_after_the_explanation(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=0).pack()), session, settings)
        await session.commit()

    assert "Задание 2 из 3" in sent.texts[-1]
    assert sent.order.index("edit") < sent.order.index("send", 1)


async def test_dont_know_is_recorded_as_skipped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§20 и §13.2: «Не знаю» — отдельный вердикт, а не неверный ответ."""
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=IDK).pack()), session, settings)
        await session.commit()

    async with sessions() as session:
        verdict = (
            await session.execute(select(SessionItem.verdict).where(SessionItem.id == item.item_id))
        ).scalar_one()
    assert verdict == "skipped"
    assert "🤷 Не знаю" in sent.edits[0]


async def test_second_tap_changes_nothing_and_says_so(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§18 и §38 №8. Первым идёт положительный ассерт, что первый тап прошёл."""
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=0).pack()), session, settings)
        await session.commit()

    async with sessions() as session:
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()
    assert logs == 1, "первый тап не прошёл — проверять нечего"

    async with sessions() as session:
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=1).pack()), session, settings)
        await session.commit()

    async with sessions() as session:
        logs_after = (
            await session.execute(select(func.count()).select_from(ReviewLog))
        ).scalar_one()
        rating = (
            await session.execute(select(SessionItem.rating).where(SessionItem.id == item.item_id))
        ).scalar_one()

    assert logs_after == 1
    assert rating == 3, "рейтинг изменён вторым нажатием"
    assert ALREADY_ANSWERED in sent.alerts
    # По одному подтверждению на каждое из двух нажатий: Telegram принимает
    # answerCallbackQuery один раз на callback_query_id, и второй вызов
    # отбросил бы именно этот алерт.
    assert sent.order.count("ack") == 2


async def test_someone_elses_item_is_refused(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Изоляция по пользователю: `si` приходит от клиента (§19.1)."""
    await _due_facts(engine)
    await scaffold(engine, tg_id=STRANGER_TG)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        await handle_answer(
            _callback(sent, Ans(si=item.item_id, v=0).pack(), tg_id=STRANGER_TG),
            session,
            settings,
        )
        await session.commit()

    async with sessions() as session:
        answered_at = (
            await session.execute(
                select(SessionItem.answered_at).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()

    assert answered_at is None
    assert logs == 0
    assert ALREADY_ANSWERED in sent.alerts
    # Одно нажатие — одно подтверждение, и это он, с алертом: Telegram
    # принимает answerCallbackQuery один раз на callback_query_id, и пустое
    # подтверждение перед алертом отбросило бы сам алерт.
    assert sent.order.count("ack") == 1


async def test_broken_question_is_refused_without_scoring(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Issue #33: строка без `correct_index` не засчитывается как неверный ответ.

    Иначе чужой `callback_data` или испорченная строка портили бы расписание
    факта через рейтинг Again (§13.2).
    """
    base = await scaffold(engine, tg_id=OWNER_TG)
    due_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    for index in range(3):
        fact_id = await add_fact(engine, base, statement=f"Факт {index}")
        await add_question(engine, fact_id, answer={"correct": 0})
        await add_review_state(engine, base.user_id, fact_id, due_at=due_at)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        await handle_answer(_callback(sent, Ans(si=item.item_id, v=0).pack()), session, settings)
        await session.commit()

    async with sessions() as session:
        answered_at = (
            await session.execute(
                select(SessionItem.answered_at).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()
        logs = (await session.execute(select(func.count()).select_from(ReviewLog))).scalar_one()

    assert answered_at is None, "испорченное задание засчитано"
    assert logs == 0
    assert BROKEN_QUESTION in sent.alerts
    assert sent.order.count("ack") == 1


async def test_summary_comes_with_the_last_explanation(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§22.1: итог печатается там же, где разбор последнего задания."""
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    for _ in range(3):
        async with sessions() as session:
            item = await current_item(session, 1)
            if item is None:
                break
            await handle_answer(
                _callback(sent, Ans(si=item.item_id, v=0).pack()), session, settings
            )
            await session.commit()

    assert "Сеанс завершён" in sent.edits[-1]
    assert "Счёт: 3 из 3" in sent.edits[-1]


def test_callback_data_fits_the_telegram_limit() -> None:
    """`callback_data` не длиннее 64 байт на предельных значениях.

    Приёмка WP-14 требует этого от всех фабрик §19.1; первая появилась здесь,
    и проверять её длину должен тест, а не надежда: aiogram длину не
    контролирует, а Telegram отвечает ошибкой отправки.
    """
    packed = Ans(si=9_223_372_036_854_775_807, v=IDK).pack()
    assert len(packed.encode("utf-8")) <= CALLBACK_DATA_LIMIT


async def test_question_text_from_the_model_is_escaped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§19.2: `html.escape` на всём, что пришло от модели.

    Незаэкранированная угловая скобка в вопросе делает сообщение невалидным
    HTML, и Telegram отказывается его отправлять — то есть задание просто не
    дойдёт.
    """
    base = await scaffold(engine, tg_id=OWNER_TG)
    due_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=5)
    payload = {
        "question": "Что значит <b>утюг</b> & пар?",
        "options": ["<мел>", "утюг & пар", "ножницы"],
    }
    for index in range(3):
        fact_id = await add_fact(engine, base, statement=f"Факт {index}")
        await add_question(engine, fact_id, payload=payload, answer=MCQ_ANSWER)
        await add_review_state(engine, base.user_id, fact_id, due_at=due_at)

    sent = Sent()
    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    text = sent.texts[0]
    assert "&lt;b&gt;утюг&lt;/b&gt;" in text
    assert "&amp; пар" in text
    assert "<b>Задание 1 из 3</b>" in text, "разметка самого бота не экранируется"


async def test_questions_shown_count_grows_once_per_delivery(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§13.3: счётчик показов растёт на выдачу, а не на команду.

    Повторный `/review` при незавершённом сеансе выдаёт то же задание и
    счётчик не растит — иначе порядок выдачи §13.3 поехал бы от нажатий.
    """
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    async with sessions() as session:
        total = (
            await session.execute(select(func.coalesce(func.sum(Question.shown_count), 0)))
        ).scalar_one()

    assert len(sent.texts) == 2, "второй /review не выдал задание"
    assert total == 1


GARBAGE_CALLBACKS = [
    ("a:", "нет полей"),
    ("a:1", "одно поле вместо двух"),
    ("a:1:2:3", "лишнее поле"),
    ("a:abc:1", "не число"),
    ("a:9999999999999999999999:0", "за пределом bigint"),
    ("a:1:9", "варианта с таким номером не бывает — §9.2"),
    ("a:0:0", "нулевой идентификатор"),
]


@pytest.mark.parametrize(("data", "reason"), GARBAGE_CALLBACKS)
def test_garbage_callback_data_is_refused_by_the_factory(data: str, reason: str) -> None:
    """Фабрика §19.1 отклоняет мусор сама, не доводя его до запроса.

    Фильтр роутера пропускает любую строку с префиксом `a:`, а `callback_data`
    приходит от клиента. Без границ в схеме `si` сверх предела `bigint`
    доходил бы до базы, где Postgres отвечает `value out of range`, —
    исключение драйвера в середине обработки выглядит сбоем системы, а не
    отклонённым вводом.
    """
    with pytest.raises((TypeError, ValueError, ValidationError)):
        Ans.unpack(data)


@pytest.mark.parametrize(("data", "reason"), GARBAGE_CALLBACKS)
async def test_garbage_callback_does_not_break_the_handler(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
    data: str,
    reason: str,
) -> None:
    """Мусорное нажатие подтверждается и объясняется, а не падает.

    §19.1: нажатие обязано быть подтверждено. Исключение из хендлера уходит
    наружу через наблюдатель `errors`, пользователь видит общий текст о сбое,
    а кнопка остаётся с часиками — то есть ровно то, от чего §19.1 защищает.
    """
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    async with sessions() as session:
        await handle_answer(_callback(sent, data), session, settings)
        await session.commit()

    assert sent.order.count("ack") == 1, f"{reason}: нажатие не подтверждено ровно один раз"
    assert BROKEN_QUESTION in sent.alerts
    assert "edit" not in sent.order, "разбор показан на неразобранном нажатии"


async def test_texts_do_not_reveal_whether_a_foreign_item_exists(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Чужое существующее задание и несуществующее отвечаются одинаково.

    Тело PR первой редакции утверждало, что три текста отказа дают один класс
    случаев, — и это было неверно: проверка ответа срабатывала раньше проверки
    владельца, поэтому на чужом **существующем** задании с испорченной строкой
    вопроса бот отвечал «не получается проверить», а на несуществующем —
    «уже отвечено». По паре ответов перебирались бы чужие идентификаторы.
    """
    base = await _due_facts(engine)
    await scaffold(engine, tg_id=STRANGER_TG)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()

    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None
        # Строка вопроса портится: на ней проверка ответа дала бы свой текст.
        await session.execute(
            update(Question).where(Question.id == item.question_id).values(answer={"correct": 0})
        )
        await session.commit()

    async with sessions() as session:
        await handle_answer(
            _callback(sent, Ans(si=item.item_id, v=0).pack(), tg_id=STRANGER_TG),
            session,
            settings,
        )
        await handle_answer(
            _callback(sent, Ans(si=10**9, v=0).pack(), tg_id=STRANGER_TG),
            session,
            settings,
        )
        await session.commit()

    assert len(sent.alerts) == 2, "один из отказов не дошёл до пользователя"
    assert (
        sent.alerts[0] == sent.alerts[1]
    ), f"по текстам различимо существование чужого задания: {sent.alerts}"
    assert base.user_id is not None


async def test_not_modified_is_ignored_but_other_failures_are_not(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§19.1: гасится ровно «message is not modified», остальное пробрасывается.

    Текст ошибки берётся **дословно тот**, что отдаёт Telegram, а не
    собирается из константы кода: фейк, построенный из той же константы, по
    которой ветвится обработчик, самосогласован при любом её значении — урок
    WP-03 про `NOT_MODIFIED`.

    Вторая половина обязательна: `RetryAfter` и «сообщение длиннее 4096» —
    сценарии отказа §37 со своим поведением, и молча гасить их нельзя.
    """
    await _due_facts(engine)
    sent = Sent()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings)
        await session.commit()
    async with sessions() as session:
        item = await current_item(session, 1)
        assert item is not None

    async def failing_edit(message: Message, error: Exception) -> None:
        async def edit_text(text: str, **kwargs: object) -> Message:
            raise error

        object.__setattr__(message, "edit_text", edit_text)

    # «message is not modified» — гасится, ответ принят.
    callback = _callback(sent, Ans(si=item.item_id, v=0).pack())
    assert isinstance(callback.message, Message)
    await failing_edit(
        callback.message,
        TelegramBadRequest(
            method=AnswerCallbackQuery(callback_query_id="x"),
            message=(
                "Telegram server says - Bad Request: message is not modified: specified "
                "new message content and reply markup are exactly the same"
            ),
        ),
    )
    async with sessions() as session:
        await handle_answer(callback, session, settings)
        await session.commit()

    async with sessions() as session:
        answered_at = (
            await session.execute(
                select(SessionItem.answered_at).where(SessionItem.id == item.item_id)
            )
        ).scalar_one()
    assert answered_at is not None, "ответ не принят, хотя правка сообщения не обязательна"

    # Любая другая ошибка Telegram — наружу, к наблюдателю `errors`.
    async with sessions() as session:
        nxt = await current_item(session, 1)
        assert nxt is not None
    other = _callback(sent, Ans(si=nxt.item_id, v=0).pack())
    assert isinstance(other.message, Message)
    await failing_edit(
        other.message,
        TelegramBadRequest(
            method=AnswerCallbackQuery(callback_query_id="x"),
            message="Telegram server says - Bad Request: message is too long",
        ),
    )
    async with sessions() as session:
        with pytest.raises(TelegramBadRequest, match="too long"):
            await handle_answer(other, session, settings)
        await session.commit()


class FakeBot:
    """Записывает правки клавиатур у сообщений, которых нет под рукой.

    В хендлер `bot` приезжает инъекцией aiogram; здесь он подставляется
    напрямую — проверяется поведение хендлера, а не диспетчера.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.cleared: list[tuple[int, int]] = []
        self.error = error

    async def edit_message_reply_markup(
        self, *, chat_id: int, message_id: int, reply_markup: object = None
    ) -> None:
        self.cleared.append((chat_id, message_id))
        if self.error is not None:
            raise self.error


async def test_second_review_leaves_one_live_keyboard(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """§20 и §3: живых клавиатур не больше одной.

    Повторный `/review` при незавершённом сеансе присылает то же задание новым
    сообщением — и обязан снять кнопки у прежнего. Нажатие на старое данных не
    изменило бы (§38 №8, это держит сервис), но два сообщения с кнопками
    выглядят сбоем, и непонятно, какое рабочее.

    Положительный ассерт первым: оба задания действительно отправлены, то есть
    прежнее сообщение существует и снимать клавиатуру есть с чего.
    """
    await _due_facts(engine)
    sent = Sent()
    bot = FakeBot()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings, bot)
        await session.commit()
    async with sessions() as session:
        control_after_first = (
            await session.execute(select(Session.control_message_id))
        ).scalar_one()

    async with sessions() as session:
        await handle_review(_message(sent), session, settings, bot)
        await session.commit()

    assert len(sent.texts) == 2, "второй /review не выдал задание"
    assert bot.cleared == [
        (OWNER_TG, control_after_first)
    ], f"клавиатура прежнего задания не снята: {bot.cleared}"

    async with sessions() as session:
        control_after_second = (
            await session.execute(select(Session.control_message_id))
        ).scalar_one()
    assert control_after_second != control_after_first, "колонка не обновилась"


async def test_deleted_previous_message_is_not_an_error(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    settings: Settings,
) -> None:
    """Прежнее сообщение удалено человеком — штатный исход, а не сбой.

    Гасятся ровно два текста Telegram: «не изменено» и «сообщение не найдено».
    Любая другая ошибка пробрасывается: снятие клавиатуры косметическое, но
    молчать о `Too Many Requests` §37 нельзя — у этого сценария своё
    поведение.
    """
    await _due_facts(engine)
    sent = Sent()

    def failing(text: str) -> FakeBot:
        return FakeBot(
            error=TelegramBadRequest(
                method=AnswerCallbackQuery(callback_query_id="x"),
                message=f"Telegram server says - {text}",
            )
        )

    async with sessions() as session:
        await handle_review(_message(sent), session, settings, FakeBot())
        await session.commit()

    async with sessions() as session:
        await handle_review(
            _message(sent), session, settings, failing("Bad Request: message to edit not found")
        )
        await session.commit()
    assert len(sent.texts) == 2, "удалённое прежнее сообщение остановило выдачу"

    async with sessions() as session:
        with pytest.raises(TelegramBadRequest, match="Too Many Requests"):
            await handle_review(
                _message(sent), session, settings, failing("Too Many Requests: retry after 5")
            )
        await session.rollback()
