"""Ответ пользователю на апдейт любого типа.

Нужен и перехватчику ошибок, и троттлингу: после ревью PR #3 оба работают на
уровне `Update` (FAIL 2.1 и 2.3), а у `Update` нет метода `answer`. Разбор
вынесен сюда, чтобы он существовал в одном месте: два экземпляра такого
разбора неизбежно разойдутся, и один из них перестанет отвечать молча.
"""

from __future__ import annotations

from aiogram.types import CallbackQuery, Message, TelegramObject, Update

from core.logging import get_logger

log = get_logger(__name__)


def respondable(event: TelegramObject) -> Message | CallbackQuery | None:
    """Объект, через который можно ответить, или `None`, если ответить некуда.

    `None` — нормальный исход, а не ошибка: на `my_chat_member` или
    `poll_answer` отвечать некому.
    """
    if isinstance(event, Update):
        return event.message or event.callback_query
    if isinstance(event, Message | CallbackQuery):
        return event
    return None


async def notify(event: TelegramObject, text: str, *, alert: bool = False) -> bool:
    """Отправляет текст пользователю. Возвращает, получилось ли.

    Сбой отправки гасится и логируется: вызывающие — перехватчик ошибок и
    троттлинг, и исключение отсюда вылетело бы наружу мимо них. Для §31 это
    означало бы падение процесса в обработчике падений.
    """
    target = respondable(event)
    if target is None:
        return False

    try:
        if isinstance(target, Message):
            await target.answer(text)
        else:
            await target.answer(text, show_alert=alert)
    except Exception:
        log.exception("failed_to_notify_user", event_type=type(event).__name__)
        return False
    return True
