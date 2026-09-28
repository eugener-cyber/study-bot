"""Команда /start и согласие на обработку данных.

ТЗ §31: согласие фиксируется при первом запуске (`users.consent_at`).

Здесь согласие показывается и нажатие обрабатывается, но в базу не пишется:
таблицы `users` до WP-02 не существует. Перенос согласован как CR-B, следы —
TODO ниже, строка в STATE.md и блок «Перенесено из WP-01» в Issue #2.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from core.logging import get_logger

log = get_logger(__name__)
router = Router(name="start")

CONSENT_CALLBACK = "consent:accept"

GREETING = (
    "Здравствуйте.\n\n"
    "Я помогаю готовиться по вашим материалам: строю конспект, составляю задания "
    "и сам возвращаю вас к теме по расписанию, пока она не закрепится.\n\n"
    "Для работы мне нужно хранить загруженные материалы, составленные по ним "
    "задания и историю ваших ответов. Всё это остаётся на вашем сервере."
)

CONSENT_ACCEPTED = "Готово. Пришлите первый материал — файлом или ссылкой."


def consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Согласен, начнём", callback_data=CONSENT_CALLBACK)]
        ]
    )


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Приветствие и запрос согласия."""
    log.info("start_command", user_id=message.from_user.id if message.from_user else None)
    await message.answer(GREETING, reply_markup=consent_keyboard())


@router.callback_query(F.data == CONSENT_CALLBACK)
async def handle_consent(callback: CallbackQuery) -> None:
    """Обрабатывает нажатие кнопки согласия.

    TODO(#2, owner: eugener-cyber): записать `users.consent_at` (ТЗ §31).
    Таблица `users` появляется в WP-02 — Issue #2, блок «Перенесено из WP-01».
    До тех пор согласие принимается, но не сохраняется; пользователи, нажавшие
    кнопку в WP-01, должны получить `consent_at` миграцией WP-02.
    """
    user_id = callback.from_user.id if callback.from_user else None
    log.info("consent_accepted", user_id=user_id)

    await callback.answer()
    if isinstance(callback.message, Message):
        await callback.message.edit_text(CONSENT_ACCEPTED)
