"""Сеанс повторения в чате: `/review` и нажатие варианта. ТЗ §19.1, §20, §22.

Хендлеры вызываются напрямую с поддельными `Message` и `CallbackQuery` — так
же, как тесты кнопок гейта в WP-03. Диспетчер здесь не поднимается: его
сборка и порядок middleware проверяются в `test_dispatcher.py`, а тут —
поведение самих хендлеров.
"""

from __future__ import annotations

import datetime

import pytest
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import func, select
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

    assert "ack" in sent.order
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
