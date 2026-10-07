"""Кнопки гейта бюджета. ТЗ §6.3, §4.5.

§6.3 задаёт три кнопки:

```
[Обработать полностью]  [Только конспект, без тестов]  [Отмена]
```

Нажатие — это и есть «продолжение ставится в очередь обработчиком нажатия
кнопки» из §4.5. Воркер к этому моменту уже завершился: он задал вопрос и
вышел, не держа слот.

`material_id` передаётся в `callback_data`, а не в состоянии FSM. Причина:
между вопросом и ответом может пройти сутки (`AWAITING_USER_TIMEOUT_H`), за
которые пользователь успеет прислать другие материалы, и состояние FSM будет
относиться к последнему из них. Идентификатор в кнопке привязан к тому
материалу, о котором спрашивали.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from arq import ArqRedis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import Material, User
from core.ingest.awaiting import awaiting_kind, resume
from core.ingest.gate import choose_full, choose_notes_only
from core.logging import get_logger
from core.services.materials import mark_failed

log = get_logger(__name__)

PREFIX = "gate"
FULL = f"{PREFIX}:full"
NOTES_ONLY = f"{PREFIX}:notes"
CANCEL = f"{PREFIX}:cancel"

ANSWER_FULL = "Обрабатываю полностью."
ANSWER_NOTES = "Сделаю только конспект. Задания можно будет достроить кнопкой."
ANSWER_CANCEL = "Отменил. Материал не обработан, файл остался."
ANSWER_STALE = "Этот вопрос уже не актуален — материал больше не ждёт ответа."
"""Ответ на устаревшее нажатие.

§38 №8: «Старый callback не изменяет данные». Кнопка остаётся в чате навсегда,
и нажатие через сутки после тайм-аута не должно ни возобновлять обработку, ни
менять режим материала.
"""


def gate_keyboard(material_id: int) -> InlineKeyboardMarkup:
    """Три кнопки §6.3. Порядок как в спецификации."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Обработать полностью", callback_data=f"{FULL}:{material_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Только конспект, без тестов",
                    callback_data=f"{NOTES_ONLY}:{material_id}",
                )
            ],
            [InlineKeyboardButton(text="Отмена", callback_data=f"{CANCEL}:{material_id}")],
        ]
    )


def _material_id(callback: CallbackQuery) -> int | None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return int(parts[2])


async def _owned_and_awaiting(
    session: AsyncSession, callback: CallbackQuery, material_id: int
) -> bool:
    """Материал принадлежит нажавшему и действительно ждёт ответа.

    Две проверки, и обе нужны. Принадлежность — изоляция по пользователю
    (чек-лист ревью, раздел 6): `callback_data` приходит от клиента, и
    подставить туда чужой `material_id` ничто не мешает. Ожидание — §38 №8:
    старый callback не изменяет данные.
    """
    owner_tg_id = (
        await session.execute(
            select(User.tg_id)
            .join(Material, Material.user_id == User.id)
            .where(Material.id == material_id)
        )
    ).scalar_one_or_none()

    if owner_tg_id != callback.from_user.id:
        log.info("gate_foreign_material", material_id=material_id, user_id=callback.from_user.id)
        return False

    return await awaiting_kind(session, material_id) == "budget"


async def handle_full(
    callback: CallbackQuery, session: AsyncSession, arq: ArqRedis | None = None
) -> None:
    """«Обработать полностью» — режим `full`, обработка продолжается."""
    await _handle(callback, session, arq, mode="full")


async def handle_notes_only(
    callback: CallbackQuery, session: AsyncSession, arq: ArqRedis | None = None
) -> None:
    """«Только конспект, без тестов» — режим `notes_only` (§6.3)."""
    await _handle(callback, session, arq, mode="notes_only")


async def handle_cancel(
    callback: CallbackQuery, session: AsyncSession, arq: ArqRedis | None = None
) -> None:
    """«Отмена» — материал в `failed` с понятной причиной.

    Причина нужна, потому что `failed` без текста выглядит сбоем системы, хотя
    пользователь сам отменил. Файл при этом не удаляется: §32 описывает
    удаление данных отдельной процедурой, и удалять его здесь значило бы
    реализовать больше, чем требует §6.3.
    """
    material_id = _material_id(callback)
    if material_id is None:
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    if not await _owned_and_awaiting(session, callback, material_id):
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    await mark_failed(session, material_id, "Обработка отменена вами. Файл сохранён.")
    await resume(session, material_id)
    await callback.answer(ANSWER_CANCEL)
    log.info("gate_cancelled", material_id=material_id)


async def _handle(
    callback: CallbackQuery,
    session: AsyncSession,
    arq: ArqRedis | None,
    *,
    mode: str,
) -> None:
    material_id = _material_id(callback)
    if material_id is None:
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    if not await _owned_and_awaiting(session, callback, material_id):
        await callback.answer(ANSWER_STALE, show_alert=True)
        return

    if mode == "notes_only":
        await choose_notes_only(session, material_id)
        answer = ANSWER_NOTES
    else:
        await choose_full(session, material_id)
        answer = ANSWER_FULL

    await resume(session, material_id)

    result = await session.execute(select(Material.origin).where(Material.id == material_id))
    origin: dict[str, object] | None = result.scalar_one()
    source_path = str((origin or {}).get("path", ""))

    if arq is not None:
        await arq.enqueue_job("process_material_task", material_id, source_path)
        log.info("gate_resumed", material_id=material_id, mode=mode)
    else:
        log.info("gate_resumed_without_worker", material_id=material_id, mode=mode)

    await callback.answer(answer)


def build_gate_router(arq: ArqRedis | None = None) -> Router:
    """Роутер кнопок гейта. Фабрика — конвенция проекта с WP-01."""
    router = Router(name="gate")

    async def on_full(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_full(callback, session, arq)

    async def on_notes(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_notes_only(callback, session, arq)

    async def on_cancel(callback: CallbackQuery, session: AsyncSession) -> None:
        await handle_cancel(callback, session, arq)

    router.callback_query.register(on_full, F.data.startswith(f"{FULL}:"))
    router.callback_query.register(on_notes, F.data.startswith(f"{NOTES_ONLY}:"))
    router.callback_query.register(on_cancel, F.data.startswith(f"{CANCEL}:"))
    return router
