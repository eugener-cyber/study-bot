"""Кнопка «Обработать заново». ТЗ §4.5, §38 №8.

§4.5 требует её прямо: «материал переводится в `failed` с понятной причиной;
повторная обработка доступна кнопкой». Пункт именно §4.5 — в ТЗ эта фраза
встречается один раз, в абзаце про тайм-аут ожидания; §4.1 перечисляет
статусы. До этого модуля текст отказа обещал кнопку, которой не было ни в
одном хендлере (Issue #44).
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.types import CallbackQuery, Chat, InaccessibleMessage, Message
from aiogram.types import User as TgUser
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bot.callbacks import BIGINT_MAX, Retry, retry_keyboard
from bot.handlers.retry import (
    ANSWER_NO_WORKER,
    ANSWER_RETRY,
    ANSWER_STALE,
    handle_retry,
)
from core.db.models import Material, User
from core.db.session import session_scope
from core.services.materials import MESSAGE_ID_KEY, progress_of

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
OWNER_TG = 2100
STRANGER_TG = 2101


async def _material(
    engine: AsyncEngine,
    *,
    status: str = "failed",
    stages: list[str] | None = None,
    tg_id: int = OWNER_TG,
) -> int:
    async with engine.begin() as connection:
        user_id = (
            await connection.execute(
                insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
            )
        ).scalar_one()
        return int(
            (
                await connection.execute(
                    insert(Material)
                    .values(
                        user_id=user_id,
                        kind="pdf",
                        status=status,
                        mode="full",
                        completed_stages=stages if stages is not None else ["extract"],
                        origin={"path": "./data/materials/x.pdf", "sha256": "a" * 64},
                        progress={},
                        error="Не удалось обработать материал.",
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


def _callback(material_id: int, tg_id: int = OWNER_TG) -> CallbackQuery:
    callback = CallbackQuery(
        id="cb-retry",
        from_user=TgUser(id=tg_id, is_bot=False, first_name="T"),
        chat_instance="x",
        data=Retry(material_id=material_id).pack(),
        message=Message(
            message_id=4242,
            date=datetime.datetime(2026, 1, 1),
            chat=Chat(id=tg_id, type="private"),
        ),
    )
    object.__setattr__(callback, "answer", AsyncMock())
    object.__setattr__(callback.message, "edit_text", AsyncMock())
    return callback


class FakeArq:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple[object, ...]]] = []

    async def enqueue_job(self, name: str, *args: object) -> None:
        self.jobs.append((name, args))


async def _state(engine: AsyncEngine, material_id: int) -> tuple[str, list[str]]:
    async with engine.connect() as connection:
        status, stages = (
            await connection.execute(
                select(Material.status, Material.completed_stages).where(Material.id == material_id)
            )
        ).one()
    return str(status), list(stages or [])


# --- Клавиатура -----------------------------------------------------------


def test_keyboard_carries_the_material_in_callback_data() -> None:
    """Кнопка одна, и она несёт идентификатор материала разобранным типом.

    Строка вида `retry:{id}` работала бы так же, но границы значения тогда
    проверял бы хендлер, а не схема: §19.1 требует валидации `callback_data`,
    и мусор должен отклоняться до запроса к базе.
    """
    rows = retry_keyboard(77).inline_keyboard

    assert len(rows) == 1 and len(rows[0]) == 1
    assert rows[0][0].text == "Обработать заново"
    assert Retry.unpack(rows[0][0].callback_data or "").material_id == 77


CALLBACK_DATA_LIMIT = 64
"""Предел Telegram на `callback_data`, в байтах.

Повторён числом, как в `tests/test_review_handlers.py`: предел задаёт
платформа, и константы в коде под него нет — проверять его должен тест, а не
надежда, потому что aiogram длину не контролирует.
"""


def test_callback_data_fits_the_telegram_limit() -> None:
    """`callback_data` не длиннее 64 байт на предельном `material_id`.

    Докстрока `bot/callbacks.py` этот предел называет, а проверяла его только
    фабрика `Ans` — замечание ревью PR #59. На `BIGINT_MAX` пакет короткий,
    запас большой, но проверяется он прогоном, а не рассуждением о длине.
    """
    packed = Retry(material_id=BIGINT_MAX).pack()

    assert len(packed.encode("utf-8")) <= CALLBACK_DATA_LIMIT


@pytest.mark.parametrize("data", ["r:0", "r:-1", "r:abc", "r:"])
def test_malformed_callback_data_is_rejected_by_the_schema(data: str) -> None:
    """Границы объявлены схемой: `pydantic` отклоняет мусор до базы."""
    with pytest.raises(Exception):  # noqa: B017 — ValidationError или ошибка разбора
        Retry.unpack(data)


# --- Нажатие --------------------------------------------------------------


async def test_retry_resumes_the_material_and_enqueues_once(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Нажатие возвращает материал в обработку и ставит задачу один раз."""
    material_id = await _material(engine)
    callback = _callback(material_id)
    arq = FakeArq()

    async with session_scope(sessions) as session:
        await handle_retry(callback, session, arq)

    status, stages = await _state(engine, material_id)
    assert status == "processing"
    assert arq.jobs == [("process_material_task", (material_id, "./data/materials/x.pdf"))]
    callback.answer.assert_awaited_once_with(ANSWER_RETRY)  # type: ignore[attr-defined]
    assert stages == ["extract"], "отметки стадий сброшены — повтор начал бы всё заново"


async def test_retry_keeps_completed_stages(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """§4.2: кнопка **возобновляет**, пройденные стадии остаются пройденными.

    Ассерт на точный список, а не на «не пусто»: сброс до непустого значения
    прошёл бы вторую проверку. Сброс всего означал бы повторные вызовы модели
    по выполненным стадиям — расход бюджета §6.3 без результата.
    """
    material_id = await _material(engine, stages=["extract", "sections", "facts"])

    async with session_scope(sessions) as session:
        await handle_retry(_callback(material_id), session, FakeArq())

    _, stages = await _state(engine, material_id)
    assert stages == ["extract", "sections", "facts"]


async def test_retry_remembers_the_message_for_progress(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """§4.5 «одно сообщение»: идентификатор запоминается при нажатии.

    Без записи прогресс следующей попытки снова оказался бы невидимым — ровно
    тот дефект, который этот тикет исправлял в PR 1a.
    """
    material_id = await _material(engine)
    callback = _callback(material_id)

    async with session_scope(sessions) as session:
        await handle_retry(callback, session, FakeArq())

    async with session_scope(sessions) as session:
        progress = await progress_of(session, material_id)

    assert progress.get(MESSAGE_ID_KEY) == 4242
    callback.message.edit_text.assert_awaited_once()  # type: ignore[union-attr]


@pytest.mark.parametrize("status", ["processing", "ready", "queued"])
async def test_retry_on_a_material_that_did_not_fail_changes_nothing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    status: str,
) -> None:
    """§38 №8: старый callback не меняет данные.

    Право на действие выдаёт условие в самом `UPDATE`, а не проверка перед
    записью: при `READ COMMITTED` между чтением и записью оставалось бы окно,
    и два нажатия подряд поставили бы две задачи.
    """
    material_id = await _material(engine, status=status)
    callback = _callback(material_id)
    arq = FakeArq()

    async with session_scope(sessions) as session:
        await handle_retry(callback, session, arq)

    assert arq.jobs == []
    assert (await _state(engine, material_id))[0] == status
    callback.answer.assert_awaited_once_with(  # type: ignore[attr-defined]
        ANSWER_STALE, show_alert=True
    )


async def test_retry_on_a_foreign_material_changes_nothing(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Чужой материал не возобновляется, и ответ тот же, что у несуществующего.

    Одинаковый ответ намеренно: разный выдавал бы факт существования чужого
    материала — утечка, которая ловилась в ревью WP-05 на оракуле ответа.

    Ассерт стоит на **самом ответе**, а не только на отсутствии задачи.
    Прежняя редакция смотрела лишь на `arq.jobs` и статусы, и мутация «чужому
    существующему — свой текст, несуществующему — другой» оставляла набор
    зелёным: свойство, объявленное в имени теста, не проверялось ничем
    (ревью PR #59).
    """
    material_id = await _material(engine)
    stranger = await _material(engine, tg_id=STRANGER_TG)
    arq = FakeArq()

    foreign = _callback(material_id, tg_id=STRANGER_TG)
    missing = _callback(999_999, tg_id=STRANGER_TG)

    async with session_scope(sessions) as session:
        await handle_retry(foreign, session, arq)
        await handle_retry(missing, session, arq)

    assert arq.jobs == []
    assert (await _state(engine, material_id))[0] == "failed"
    assert (await _state(engine, stranger))[0] == "failed"

    foreign.answer.assert_awaited_once_with(ANSWER_STALE, show_alert=True)  # type: ignore[attr-defined]
    missing.answer.assert_awaited_once_with(ANSWER_STALE, show_alert=True)  # type: ignore[attr-defined]
    assert (
        foreign.answer.await_args == missing.answer.await_args  # type: ignore[attr-defined]
    ), "ответы различаются — наблюдаемая разница выдаёт факт существования материала"


async def test_retry_answers_honestly_when_there_is_no_worker(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Без пула ARQ человек слышит отказ, а не «обрабатываю заново».

    Ветвь в продакшене недостижима — `bot/main.py` всегда строит пул, — но
    ответ «обрабатываю» при материале в `processing`, который никто не
    обрабатывает, обещал бы работу, которой не будет. Замечание ревью PR #59.
    """
    material_id = await _material(engine)
    callback = _callback(material_id)

    async with session_scope(sessions) as session:
        await handle_retry(callback, session, None)

    callback.answer.assert_awaited_once_with(  # type: ignore[attr-defined]
        ANSWER_NO_WORKER, show_alert=True
    )


async def test_retry_survives_an_inaccessible_message(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Сообщение старше 48 часов приезжает `InaccessibleMessage` — и это не сбой.

    У него нет ни `edit_text`, ни смысла правиться, но задача обязана уйти в
    очередь: кнопка нажата человеком, который ждёт обработки, а не правки
    текста. `message_id` при этом не запоминается — его неоткуда взять.
    """
    material_id = await _material(engine)
    callback = _callback(material_id)
    object.__setattr__(
        callback,
        "message",
        InaccessibleMessage(message_id=0, chat=Chat(id=OWNER_TG, type="private")),
    )
    arq = FakeArq()

    async with session_scope(sessions) as session:
        await handle_retry(callback, session, arq)

    assert arq.jobs == [("process_material_task", (material_id, "./data/materials/x.pdf"))]
    callback.answer.assert_awaited_once_with(ANSWER_RETRY)  # type: ignore[attr-defined]

    async with session_scope(sessions) as session:
        assert MESSAGE_ID_KEY not in await progress_of(session, material_id)


async def test_second_tap_does_not_enqueue_twice(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Двойное нажатие даёт одну задачу: второе условие `UPDATE` не выполнится.

    Нажатия идут одно за другим в одной транзакции — это проверяет
    идемпотентность по статусу, но **не** отсутствие окна между чтением и
    записью: перекрытие транзакций нужно проверять отдельно.
    """
    material_id = await _material(engine)
    arq = FakeArq()

    async with session_scope(sessions) as session:
        await handle_retry(_callback(material_id), session, arq)
        await handle_retry(_callback(material_id), session, arq)

    assert len(arq.jobs) == 1
