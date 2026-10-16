"""Повторная обработка материала после отказа. ТЗ §4.1, §4.5.

§4.1 требует кнопку прямо: «материал переводится в `failed` с понятной
причиной; **повторная обработка доступна кнопкой**». До этого модуля текст
отказа обещал кнопку, которой не было ни в одном хендлере (Issue #44).

Кнопка **возобновляет**, а не начинает заново. §4.2: «при исключении до COMMIT
`completed_stages` не меняется, следующая попытка начинает стадию заново» —
то есть пройденные стадии остаются пройденными, а повтор продолжает с
упавшей. Сброс всего означал бы повторные вызовы модели по уже выполненным
стадиям, то есть расход бюджета §6.3 без результата.
"""

from __future__ import annotations

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from arq import ArqRedis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bot.callbacks import Retry
from core.db.models import Material, User
from core.logging import get_logger
from core.services.materials import MESSAGE_ID_KEY, progress_of

log = get_logger(__name__)

ANSWER_RETRY = "Обрабатываю заново."
ANSWER_STALE = "Это сообщение устарело: материал уже обрабатывается или готов."
ANSWER_FOREIGN = "Этот материал не ваш."
PROGRESS_TEXT = "Обрабатываю заново…"


def retry_keyboard(material_id: int) -> InlineKeyboardMarkup:
    """Кнопка под сообщением об отказе. §4.1.

    Собирается здесь, а не в воркере: воркеру нужна разметка, но знать про
    хендлеры он не должен — он импортирует только фабрику `callback_data`.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Обработать заново",
                    callback_data=Retry(material_id=material_id).pack(),
                )
            ]
        ]
    )


async def handle_retry(
    callback: CallbackQuery,
    session: AsyncSession,
    arq: ArqRedis | None = None,
) -> None:
    """Нажатие кнопки «Обработать заново».

    **Право на действие выдаёт база**, а не проверка перед записью: условный
    `UPDATE … WHERE status='failed' AND user_id=… RETURNING` либо меняет
    строку, либо нет, и второй одновременный тап получает пустой результат.
    Проверка «статус failed?» отдельным запросом оставила бы окно между
    чтением и записью — при `READ COMMITTED` это ровно тот дефект, который
    ловился в WP-05 на двойном нажатии ответа.

    Владелец сверяется **в том же** `UPDATE`: подставить чужой `material_id`
    в `callback_data` ничто не мешает, а отдельный запрос про владельца дал бы
    разный наблюдаемый ответ для «чужого существующего» и «несуществующего»
    материала — утечка факта существования, которая ловилась в ревью WP-05.
    """
    data = Retry.unpack(callback.data or "")
    material_id = data.material_id

    owner_tg_id = callback.from_user.id
    claimed = (
        await session.execute(
            update(Material)
            .where(
                Material.id == material_id,
                Material.status == "failed",
                Material.user_id.in_(select(User.id).where(User.tg_id == owner_tg_id)),
            )
            .values(status="processing", error=None)
            .returning(Material.id, Material.origin)
        )
    ).first()

    if claimed is None:
        log.info("retry_rejected", material_id=material_id, user_id=owner_tg_id)
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    _, origin = claimed
    source_path = str((origin or {}).get("path", ""))

    message = await _refresh_message(callback, session, material_id)

    if arq is not None:
        await arq.enqueue_job("process_material_task", material_id, source_path)
        log.info("retry_enqueued", material_id=material_id, message_id=message)
    else:
        log.info("retry_without_worker", material_id=material_id)

    await callback.answer(ANSWER_RETRY)


async def _refresh_message(
    callback: CallbackQuery, session: AsyncSession, material_id: int
) -> int | None:
    """Правит сообщение на «обрабатываю заново» и запоминает его идентификатор.

    Запомнить обязательно: §4.5 требует одно сообщение на обработку, и без
    записи `message_id` прогресс следующей попытки снова оказался бы
    невидимым — ровно тот дефект, который этот тикет и исправлял в PR 1a.

    Гасится **только** ошибка API Telegram (`TelegramAPIError`): §4.5 про
    отображение, и падать из-за косметики, когда задача уже поставлена,
    значило бы показать человеку ошибку вместо начавшейся обработки. Широкий
    `except Exception` здесь был бы п. 4 списка костылей — он спрятал бы и
    ошибку записи `message_id`, и опечатку в разметке.
    """
    message = callback.message
    message_id = getattr(message, "message_id", None)

    if message_id is not None:
        progress = await progress_of(session, material_id)
        progress[MESSAGE_ID_KEY] = int(message_id)
        await session.execute(
            update(Material).where(Material.id == material_id).values(progress=progress)
        )

    edit = getattr(message, "edit_text", None)
    if edit is not None:
        try:
            await edit(PROGRESS_TEXT, reply_markup=None)
        except TelegramAPIError as error:
            log.info(
                "retry_message_not_edited",
                material_id=material_id,
                error_type=type(error).__name__,
            )

    return int(message_id) if message_id is not None else None


def build_retry_router(arq: ArqRedis | None = None) -> Router:
    """Роутер кнопки повтора. Фабрика — конвенция проекта с WP-01."""
    router = Router(name="retry")

    async def on_retry(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_retry(callback, session, arq)

    router.callback_query.register(on_retry, Retry.filter())
    return router
