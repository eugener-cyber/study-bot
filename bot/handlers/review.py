"""Сеанс повторения в чате: команда `/review` и ответ на задание.

ТЗ §19.1, §19.2, §20, §22, §22.1. Хендлеры здесь только разбирают апдейт,
вызывают сервис и рендерят — §конвенция проекта; вся логика сеанса живёт в
`core/services/sessions.py` и `core/learning/*`, и консольный путь §1.4
проверяет ровно её.

**Чего в разметке нет и почему.** §20 для MCQ требует вопрос и
пронумерованные варианты в тексте плюс клавиатуру `1 2 3 4` и `🤷 Не знаю` —
это сделано. Кнопки `📍 Показать в источнике` и `[⚠️]` из §22 не рисуются:
переход к источнику требует настоящих локаторов (WP-06) и рендера страниц
(WP-11), а жалобы — это §24 и WP-17. Кнопки, которой не на что нажимать, в
интерфейсе быть не должно.

`answer_callback_query` вызывается **до** правки сообщения (§19.1): иначе при
медленной правке Telegram показывает часики на кнопке, а при ошибке правки
нажатие остаётся неподтверждённым вовсе.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bot.callbacks import IDK, Ans
from core.config import Settings
from core.db.models import User
from core.learning.checking import InvalidAnswerError
from core.learning.explain import Explanation
from core.learning.scheduler import FixedIntervalScheduler
from core.logging import get_logger
from core.services.sessions import (
    AnswerResult,
    ItemView,
    SessionSummary,
    answer_item,
    assemble,
    current_item,
    mark_shown,
    set_control_message,
)

log = get_logger(__name__)

NOTHING_TO_REVIEW = (
    "Повторять пока нечего. Задания приходят, когда наступает срок повторения — "
    "обычно через десять минут после обработки материала."
)
"""§16: «Если после всех правил кандидатов меньше MIN_SESSION_ITEMS, сессия не
создаётся и факты ждут следующего тика». Молчать в ответ на команду нельзя —
это ровно тот дефект, из-за которого в WP-03 появился §19.4."""

ALREADY_ANSWERED = "Это задание уже отвечено."
"""§18 и §38 №8: двойной тап и старый callback. Текст один на все три случая —
сессия не активна, задание отвечено, задание чужое: по разным ответам можно
было бы перебирать идентификаторы чужих заданий."""

BROKEN_QUESTION = "Это задание не получается проверить. Оно не засчитано."
"""`InvalidAnswerError`: испорченная строка вопроса (Issue #33) или подделанный
`callback_data`. Ответ не засчитывается — §13.2 превратил бы «неверно» в
рейтинг Again и испортил расписание факта по чужому запросу."""

IDK_BUTTON = "🤷 Не знаю"
"""§20. Для MCQ показывается сразу — отложенные кнопки (CR-4,
`GIVEUP_DELAY_SEC`) относятся к открытому ответу, то есть к WP-14."""


def task_keyboard(item: ItemView) -> InlineKeyboardMarkup:
    """Клавиатура MCQ: номера одним рядом плюс «Не знаю». §20."""
    options = list(item.payload.get("options") or [])
    numbers = [
        InlineKeyboardButton(
            text=str(number),
            callback_data=Ans(si=item.item_id, v=index).pack(),
        )
        for index, number in enumerate(range(1, len(options) + 1))
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            numbers,
            [
                InlineKeyboardButton(
                    text=IDK_BUTTON,
                    callback_data=Ans(si=item.item_id, v=IDK).pack(),
                )
            ],
        ]
    )


def render_task(item: ItemView) -> str:
    """Текст задания. §19.2: `parse_mode=HTML` и `html.escape` на всём от модели."""
    options = list(item.payload.get("options") or [])
    lines = [
        f"<b>Задание {item.position} из {item.total}</b>",
        "",
        html.escape(str(item.payload.get("question", ""))),
        "",
    ]
    lines.extend(
        f"{number}. {html.escape(str(text))}" for number, text in enumerate(options, start=1)
    )
    return "\n".join(lines)


VERDICT_LABEL = {
    "correct": "✅ Верно",
    "wrong": "❌ Неверно",
    "skipped": "🤷 Не знаю",
    "partial": "◐ Частично",
}
"""Подписи §22. `partial` приедет с открытым ответом в WP-15; подпись заведена
здесь, чтобы разбор не падал на незнакомом вердикте."""


def render_explanation(explanation: Explanation) -> str:
    """Разбор §22. Модель не вызывается: `detail` пришёл вместе с фактом."""
    lines = [
        VERDICT_LABEL.get(explanation.verdict, explanation.verdict),
        f"Правильный ответ: {html.escape(explanation.correct_answer)}",
        "",
        f"<b>Почему:</b> {html.escape(explanation.detail)}",
    ]
    lines.extend(f"<i>В источнике:</i> {html.escape(hint)}" for hint in explanation.locator_hints)
    return "\n".join(lines)


def render_summary(summary: SessionSummary) -> str:
    """Итог сеанса §22.1: счёт, время, список ошибок.

    Обновлённой mastery затронутых тем и кнопок «Повторить ошибки сейчас» и
    «К конспекту» здесь нет: mastery — WP-16, быстрое повторение — §27 и
    WP-17, конспект — §8.1 и WP-08.
    """
    minutes, seconds = divmod(summary.seconds, 60)
    spent = f"{minutes} мин {seconds} с" if minutes else f"{seconds} с"
    lines = [
        "<b>Сеанс завершён</b>",
        f"Счёт: {summary.correct} из {summary.total}",
        f"Время: {spent}",
    ]
    if summary.mistakes:
        lines.append("")
        lines.append("<b>Ошибки:</b>")
        lines.extend(f"— {html.escape(statement)}" for statement in summary.mistakes)
    return "\n".join(lines)


async def _user_id(session: AsyncSession, tg_id: int) -> int | None:
    """Внутренний идентификатор по Telegram id.

    `None` означает, что человек ещё не давал согласия: строка `users`
    создаётся в момент согласия (§31, решение WP-02), и до него расписания у
    него нет по построению.
    """
    found = (await session.execute(select(User.id).where(User.tg_id == tg_id))).scalar_one_or_none()
    return int(found) if found is not None else None


async def handle_review(message: Message, session: AsyncSession, settings: Settings) -> None:
    """Команда `/review`: собрать или продолжить сеанс и выдать задание.

    При незавершённом сеансе выдаётся **текущее** задание, а не новое: §18
    допускает одну активную сессию, и счётчик показов при повторной выдаче не
    растёт (§13.3) — об этом заботится `mark_shown`.
    """
    if message.from_user is None:  # pragma: no cover — у команды автор есть всегда
        return

    user_id = await _user_id(session, message.from_user.id)
    if user_id is None:
        await message.answer(NOTHING_TO_REVIEW)
        return

    now = datetime.now(UTC)
    session_id = await assemble(session, user_id=user_id, settings=settings, now=now)
    if session_id is None:
        await message.answer(NOTHING_TO_REVIEW)
        return

    item = await current_item(session, session_id)
    if item is None:  # pragma: no cover — активная сессия без неотвеченных заданий
        await message.answer(NOTHING_TO_REVIEW)
        return

    await mark_shown(session, item.item_id, user_id=user_id, now=now)
    await _send_task(message, session, item)
    log.info("review_task_sent", user_id=user_id, session_id=session_id, item=item.item_id)


async def handle_answer(callback: CallbackQuery, session: AsyncSession, settings: Settings) -> None:
    """Нажатие варианта или «Не знаю».

    Порядок строгий: подтверждение нажатия (§19.1) → мутация в сервисе →
    правка сообщения с разбором → следующее задание. Подтверждение первым
    потому, что правка может не удаться, а неподтверждённое нажатие Telegram
    показывает часиками на кнопке.
    """
    if callback.data is None or callback.from_user is None:  # pragma: no cover
        return

    parsed = Ans.unpack(callback.data)
    await callback.answer()

    user_id = await _user_id(session, callback.from_user.id)
    if user_id is None:
        await callback.answer(ALREADY_ANSWERED, show_alert=True)
        return

    raw: dict[str, object] = {"idk": True} if parsed.v == IDK else {"option": parsed.v}
    now = datetime.now(UTC)

    try:
        result = await answer_item(
            session,
            user_id=user_id,
            item_id=parsed.si,
            raw=raw,
            now=now,
            settings=settings,
            scheduler=FixedIntervalScheduler(),
        )
    except InvalidAnswerError as error:
        log.info("answer_not_parsed", item_id=parsed.si, reason=str(error))
        await callback.answer(BROKEN_QUESTION, show_alert=True)
        return

    if result is None:
        await callback.answer(ALREADY_ANSWERED, show_alert=True)
        return

    await _show_explanation(callback, result)

    if result.finished or not isinstance(callback.message, Message):
        return

    session_id = await _session_of(session, parsed.si)
    if session_id is None:  # pragma: no cover — элемент только что отвечен
        return
    item = await current_item(session, session_id)
    if item is None:  # pragma: no cover — сессия уже завершена выше
        return

    await mark_shown(session, item.item_id, user_id=user_id, now=now)
    await _send_task(callback.message, session, item)


async def _send_task(message: Message, session: AsyncSession, item: ItemView) -> None:
    """Отправляет задание и запоминает сообщение с клавиатурой (§20).

    `sessions.control_message_id` — колонка §3 именно под это: по завершении
    задания клавиатуру нужно убрать, а убрать её можно только у известного
    сообщения.
    """
    sent = await message.answer(
        render_task(item), reply_markup=task_keyboard(item), parse_mode="HTML"
    )
    if isinstance(sent, Message):
        await set_control_message(session, item.session_id, sent.message_id)


async def _show_explanation(callback: CallbackQuery, result: AnswerResult) -> None:
    """Правит сообщение задания разбором и снимает клавиатуру.

    Клавиатура снимается, а не оставляется: §38 №8 требует, чтобы старый
    callback не менял данные, и сервис это обеспечивает, — но кнопка, которая
    ничего не делает, выглядит сбоем. Сбой правки гасится: ответ уже записан,
    и падение здесь означало бы «ответ не принят» при принятом ответе.
    """
    text = render_explanation(result.explanation)
    if result.summary is not None:
        text = f"{text}\n\n{render_summary(result.summary)}"

    if not isinstance(callback.message, Message):
        return
    try:
        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=None)
    except Exception:
        log.info("explanation_not_edited", item_id=result.explanation.statement[:40])


async def _session_of(session: AsyncSession, item_id: int) -> int | None:
    from core.db.models import SessionItem

    found = (
        await session.execute(select(SessionItem.session_id).where(SessionItem.id == item_id))
    ).scalar_one_or_none()
    return int(found) if found is not None else None


def build_review_router(settings: Settings) -> Router:
    """Роутер сеанса повторения. Фабрика — конвенция проекта с WP-01."""
    router = Router(name="review")

    async def on_review(message: Message, session: AsyncSession) -> None:
        await handle_review(message, session, settings)

    async def on_answer(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_answer(callback, session, settings)

    router.message.register(on_review, Command("review"))
    router.callback_query.register(on_answer, F.data.startswith(f"{Ans.__prefix__}:"))
    return router
