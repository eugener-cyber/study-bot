"""Повторная обработка материала после отказа. ТЗ §4.5, §35.

§4.5 требует кнопку прямо: «материал переводится в `failed` с понятной
причиной; **повторная обработка доступна кнопкой**». Пункт именно §4.5: в ТЗ
эта фраза встречается один раз, в абзаце про тайм-аут ожидания. До этого
модуля текст отказа обещал кнопку, которой не было ни в одном хендлере
(Issue #44).

Кнопка **возобновляет**, а не начинает заново. §4.2: «при исключении до COMMIT
`completed_stages` не меняется, следующая попытка начинает стадию заново» —
то есть пройденные стадии остаются пройденными, а повтор продолжает с
упавшей. Сброс всего означал бы повторные вызовы модели по уже выполненным
стадиям, то есть расход бюджета §6.3 без результата.

§35: здесь только разбор апдейта, вызов сервиса и ответ. Переход статуса
живёт в `core/services/materials.claim_failed_for_retry`, запись `message_id` —
в `core/ingest/awaiting.remember_message`. Первая редакция выполняла и то и
другое `update(Material)` прямо здесь, причём вторая половина повторяла
существующий сервис строка в строку (ревью PR #59).
"""

from __future__ import annotations

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, Message
from arq import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from bot.callbacks import Retry
from core.ingest.awaiting import remember_message
from core.logging import get_logger
from core.services.materials import claim_failed_for_retry

log = get_logger(__name__)

ANSWER_RETRY = "Обрабатываю заново."
ANSWER_STALE = "Это сообщение устарело: материал уже обрабатывается или готов."
"""Ответ на любой отказ — и на чужой материал, и на несуществующий.

Один текст на все причины, а не «этот материал не ваш»: подставить чужой
`material_id` в `callback_data` ничто не мешает, и разный ответ выдал бы факт
существования материала. Отдельная константа под чужой материал в первой
редакции лежала в модуле неиспользованной — мёртвый код, который к тому же
создавал бы утечку, если бы его кто-нибудь применил (ревью PR #59).
"""
ANSWER_NO_WORKER = "Повтор не запущен: обработчик недоступен. Попробуйте позже."
"""Воркера нет — и человек слышит об этом, а не «обрабатываю заново».

Ветвь в продакшене недостижима (`bot/main.py` всегда строит пул), но ответ
«обрабатываю» при материале в `processing`, который никто не обрабатывает,
обещал бы работу, которой не будет: §4.1 такого состояния не предусматривает.
`bot/handlers/upload.py` в той же ситуации отвечает честно — здесь так же.
"""
PROGRESS_TEXT = "Обрабатываю заново…"


async def handle_retry(
    callback: CallbackQuery,
    session: AsyncSession,
    arq: ArqRedis | None = None,
) -> None:
    """Нажатие кнопки «Обработать заново».

    Право на действие выдаёт сервис одним условным `UPDATE` — там же, в его
    докстроке, названо, почему проверка перед записью оставила бы окно между
    чтением и записью и почему причина отказа наружу не возвращается.
    """
    data = Retry.unpack(callback.data or "")
    material_id = data.material_id

    origin = await claim_failed_for_retry(session, material_id, callback.from_user.id)

    if origin is None:
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    source_path = str(origin.get("path", ""))
    message = await _refresh_message(callback, session, material_id)

    if arq is None:
        log.warning("retry_without_worker", material_id=material_id)
        await callback.answer(ANSWER_NO_WORKER, show_alert=True)
        return

    await arq.enqueue_job("process_material_task", material_id, source_path)
    log.info("retry_enqueued", material_id=material_id, message_id=message)
    await callback.answer(ANSWER_RETRY)


async def _refresh_message(
    callback: CallbackQuery, session: AsyncSession, material_id: int
) -> int | None:
    """Правит сообщение на «обрабатываю заново» и запоминает его идентификатор.

    Запомнить обязательно: §4.5 требует одно сообщение на обработку, и без
    записи `message_id` прогресс следующей попытки снова оказался бы
    невидимым — ровно тот дефект, который этот тикет и исправлял в PR 1a.

    `isinstance(message, Message)` сужает тип, а не `getattr` по имени поля:
    у нажатия может приехать `InaccessibleMessage` (сообщение старше 48 часов
    удалено на стороне Telegram), и у него нет ни `edit_text`, ни смысла
    правиться. Первая редакция обходила это `getattr(..., None)`, то есть
    проходила мимо `mypy --strict` без объяснения (п. 6 списка костылей).

    Гасится **только** ошибка API Telegram (`TelegramAPIError`): §4.5 про
    отображение, и падать из-за косметики, когда задача уже поставлена,
    значило бы показать человеку ошибку вместо начавшейся обработки. Широкий
    `except Exception` здесь был бы п. 4 списка костылей — он спрятал бы и
    ошибку записи `message_id`, и опечатку в разметке.
    """
    message = callback.message
    if not isinstance(message, Message):
        log.info(
            "retry_message_unavailable",
            material_id=material_id,
            message_type=type(message).__name__,
        )
        return None

    await remember_message(session, material_id, message.message_id)

    try:
        await message.edit_text(PROGRESS_TEXT, reply_markup=None)
    except TelegramAPIError as error:
        log.info(
            "retry_message_not_edited",
            material_id=material_id,
            error_type=type(error).__name__,
        )

    return message.message_id


def build_retry_router(arq: ArqRedis | None = None) -> Router:
    """Роутер кнопки повтора. Фабрика — конвенция проекта с WP-01."""
    router = Router(name="retry")

    async def on_retry(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_retry(callback, session, arq)

    router.callback_query.register(on_retry, Retry.filter())
    return router
