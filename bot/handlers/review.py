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

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from bot.callbacks import IDK, Ans
from bot.progress import NOT_MODIFIED
from core.config import Settings
from core.learning.checking import InvalidAnswerError
from core.learning.explain import Explanation, format_duration
from core.learning.scheduler import FixedIntervalScheduler
from core.logging import get_logger
from core.services.sessions import (
    AnswerResult,
    ItemView,
    SessionSummary,
    answer_item,
    assemble,
    control_message_id,
    current_item,
    mark_shown,
    session_of_item,
    set_control_message,
)
from core.services.users import find_by_tg_id

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

MESSAGE_GONE = "message to edit not found"
"""Фрагмент ответа Telegram, когда правимое сообщение удалено.

Снятие клавиатуры у прежнего задания — косметика: человек мог удалить
сообщение сам, и это нормальный исход, а не сбой. Как и `NOT_MODIFIED`,
различается по тексту — отдельного класса исключения в aiogram 3 нет.
"""

IDK_BUTTON = "🤷 Не знаю"
"""§20. Для MCQ показывается сразу — отложенные кнопки (CR-4,
`GIVEUP_DELAY_SEC`) относятся к открытому ответу, то есть к WP-14."""


def task_keyboard(item: ItemView) -> InlineKeyboardMarkup:
    """Клавиатура MCQ: номера одним рядом плюс «Не знаю». §20."""
    options = list(item.payload.get("options") or [])
    rows = []
    if options:
        # Пустой ряд Telegram отклоняет, и задание вообще не дошло бы. §9.2
        # требует от mcq двух-четырёх вариантов, то есть строка без них
        # испорчена (Issue #33); ответить на неё можно только «не знаю», и
        # это честнее, чем отправка, падающая на стороне Telegram.
        rows.append(
            [
                InlineKeyboardButton(
                    text=str(number),
                    callback_data=Ans(si=item.item_id, v=number - 1).pack(),
                )
                for number in range(1, len(options) + 1)
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text=IDK_BUTTON,
                callback_data=Ans(si=item.item_id, v=IDK).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


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
    lines = [
        "<b>Сеанс завершён</b>",
        f"Счёт: {summary.correct} из {summary.total}",
        f"Время: {format_duration(summary.seconds)}",
    ]
    if summary.mistakes:
        lines.append("")
        lines.append("<b>Ошибки:</b>")
        lines.extend(f"— {html.escape(statement)}" for statement in summary.mistakes)
    return "\n".join(lines)


async def _user_id(session: AsyncSession, tg_id: int) -> int | None:
    """Внутренний идентификатор по Telegram id.

    Через сервис, а не запросом здесь: §35 и конвенция проекта не дают
    хендлерам читать таблицы напрямую, а `find_by_tg_id` уже существует и уже
    вызывается приёмом файла (`bot/handlers/upload.py`). Третья копия той же
    выборки — п. 5 списка костылей.

    `None` означает, что человек ещё не давал согласия: строка `users`
    создаётся в момент согласия (§31, решение WP-02), и до него расписания у
    него нет по построению.
    """
    user = await find_by_tg_id(session, tg_id)
    return int(user.id) if user is not None else None


async def handle_review(
    message: Message,
    session: AsyncSession,
    settings: Settings,
    bot: Bot,
) -> None:
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
    await _send_task(message, session, item, bot)
    log.info("review_task_sent", user_id=user_id, session_id=session_id, item=item.item_id)


async def handle_answer(
    callback: CallbackQuery,
    session: AsyncSession,
    settings: Settings,
    bot: Bot,
) -> None:
    """Нажатие варианта или «Не знаю».

    Порядок строгий: мутация в сервисе → подтверждение нажатия (§19.1) →
    правка сообщения с разбором → следующее задание. Подтверждение стоит
    **до правки**, потому что правка может не удаться, а неподтверждённое
    нажатие Telegram показывает часиками на кнопке.

    **Подтверждение ровно одно на нажатие.** Первая редакция подтверждала
    дважды — пустым вызовом сразу после разбора `callback_data` и затем
    алертом в ветках отказа, — а Telegram принимает `answerCallbackQuery`
    один раз на `callback_query_id`: второй отбрасывается, то есть тексты
    §18 и §38 №8 до человека не доходили бы вовсе. Замечание 6 ревью PR #35.
    Фейк в тестах этого не показывал, потому что всегда отвечал успехом;
    теперь тесты считают подтверждения.
    """
    if callback.data is None or callback.from_user is None:  # pragma: no cover
        return

    try:
        parsed = Ans.unpack(callback.data)
    except (TypeError, ValueError, ValidationError) as error:
        # Фильтр роутера пропускает любую строку, начинающуюся с префикса
        # фабрики, а `callback_data` приходит от клиента (§19.1). Разбор
        # обязан отклонять такой ввод, а не падать: исключение здесь выходит
        # наружу через наблюдатель `errors` и выглядит сбоем системы, причём
        # нажатие остаётся неподтверждённым — §19.1 требует обратного.
        log.info("callback_not_parsed", data=callback.data, reason=type(error).__name__)
        await callback.answer(BROKEN_QUESTION, show_alert=True)
        return

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

    # Единственное подтверждение успешного пути — и оно до правки сообщения.
    await callback.answer()
    await _show_explanation(callback, result, item_id=parsed.si)

    if result.finished or not isinstance(callback.message, Message):
        return

    session_id = await session_of_item(session, parsed.si, user_id=user_id)
    if session_id is None:  # pragma: no cover — элемент только что отвечен
        return
    item = await current_item(session, session_id)
    if item is None:  # pragma: no cover — сессия уже завершена выше
        return

    await mark_shown(session, item.item_id, user_id=user_id, now=now)
    await _send_task(
        callback.message,
        session,
        item,
        bot,
        already_cleared=callback.message.message_id,
    )


async def _send_task(
    message: Message,
    session: AsyncSession,
    item: ItemView,
    bot: Bot,
    *,
    already_cleared: int | None = None,
) -> None:
    """Отправляет задание, снимая клавиатуру у прежнего сообщения. §20, §3.

    `sessions.control_message_id` существует ровно для этого: живых клавиатур
    должно быть не больше одной. Иначе второй `/review` при незавершённом
    сеансе оставляет два сообщения с кнопками, нажатие на старое данных не
    меняет (§38 №8, это держит сервис) — но выглядит как сбой, и человек не
    понимает, какое из сообщений рабочее.

    Первая редакция колонку **писала и не читала**, а две докстроки утверждали
    обратное — замечание 5 ревью PR #35. Запись без чтения не закрывает ничего.

    Порядок: сначала новое сообщение, потом снятие клавиатуры у старого. При
    обратном порядке неудачная отправка оставила бы человека вообще без
    кнопок. Два исхода правки игнорируются как штатные — «не изменено» и
    «сообщение не найдено», — остальные пробрасываются.

    `already_cleared` — сообщение, с которого клавиатуру уже сняли в этом же
    обработчике. На пути ответа это то самое сообщение, которое стало
    разбором: без оговорки сюда шёл второй вызов API на каждый ответ, Telegram
    отвечал «message is not modified», и в журнал писалась строка о неснятой
    клавиатуре — на совершенно нормальном пути.
    """
    previous = await control_message_id(session, item.session_id)

    sent = await message.answer(
        render_task(item), reply_markup=task_keyboard(item), parse_mode="HTML"
    )
    if isinstance(sent, Message):
        await set_control_message(session, item.session_id, sent.message_id)

    if (
        previous is None
        or (isinstance(sent, Message) and previous == sent.message_id)
        # Клавиатуру этого сообщения уже снял разбор: второй вызов API получил
        # бы «message is not modified» и писал бы в журнал строку о неснятой
        # клавиатуре на совершенно нормальном пути.
        or previous == already_cleared
    ):
        return
    try:
        await bot.edit_message_reply_markup(
            chat_id=message.chat.id, message_id=previous, reply_markup=None
        )
    except TelegramBadRequest as error:
        if NOT_MODIFIED not in str(error) and MESSAGE_GONE not in str(error):
            raise
        log.info("previous_keyboard_not_cleared", message_id=previous)


async def _show_explanation(callback: CallbackQuery, result: AnswerResult, *, item_id: int) -> None:
    """Правит сообщение задания разбором и снимает клавиатуру.

    Клавиатура снимается, а не оставляется: §38 №8 требует, чтобы старый
    callback не менял данные, и сервис это обеспечивает, — но кнопка, которая
    ничего не делает, выглядит сбоем.

    **Гасится ровно одно исключение** — «message is not modified» (§19.1):
    оно неизбежно при быстрых повторных нажатиях и означает, что текст уже
    такой. Остальные пробрасываются. Первая редакция ловила `Exception`, и
    вместе с этим случаем молча гасились `RetryAfter` и «сообщение длиннее
    4096» — оба §37, то есть сценарии отказа со своим поведением, — и ошибки
    разбора HTML, то есть дефект §19.2 стал бы невидимым (п. 4 списка
    костылей). Константа переиспользуется из `bot/progress.py`: отдельного
    класса исключения в aiogram 3 нет, различение по тексту, и двух копий
    этого текста быть не должно.
    """
    text = render_explanation(result.explanation)
    if result.summary is not None:
        text = f"{text}\n\n{render_summary(result.summary)}"

    if not isinstance(callback.message, Message):
        return
    try:
        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=None)
    except TelegramBadRequest as error:
        if NOT_MODIFIED not in str(error):
            raise
        log.info("explanation_not_modified", item_id=item_id)


def build_review_router(settings: Settings) -> Router:
    """Роутер сеанса повторения. Фабрика — конвенция проекта с WP-01."""
    router = Router(name="review")

    async def on_review(message: Message, session: AsyncSession, bot: Bot) -> None:
        await handle_review(message, session, settings, bot)

    async def on_answer(callback: CallbackQuery, session: AsyncSession, bot: Bot) -> None:
        await handle_answer(callback, session, settings, bot)

    router.message.register(on_review, Command("review"))
    router.callback_query.register(on_answer, F.data.startswith(f"{Ans.__prefix__}:"))
    return router
