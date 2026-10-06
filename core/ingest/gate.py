"""Шаг `probe` и гейт бюджета перед машиной стадий. ТЗ §6.3.

`probe` вынесен **перед** конвейером, а не сделан первой стадией. Причина
техническая: стадии пишутся в `completed_stages` и после выполнения не
повторяются (§4.3), а `probe` должен переисполняться при каждой попытке —
он ничего не стоит и ничего не сохраняет, кроме оценки.

Причина по существу — §6.3: оценка считается до `extract`, потому что для
сканированного PDF `extract` это vision-вызов на каждую страницу, то есть
основная статья расхода. Гейт после него спрашивал бы «обрабатывать?» после
того, как деньги потрачены.
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.adapters.base import SourceAdapter
from core.config import Settings
from core.db.models import Fragment, Material
from core.db.session import session_scope
from core.ingest.awaiting import park
from core.ingest.budget import (
    CostEstimate,
    estimate_after_extract,
    estimate_from_probe,
    exceeds_threshold,
    gate_question,
    needs_reestimate_confirmation,
)
from core.ingest.pipeline import AwaitingUser
from core.logging import get_logger

log = get_logger(__name__)

GATE_KIND = "budget"
"""Значение `progress.kind` при ожидании подтверждения бюджета (§4.5)."""


async def probe_and_gate(
    sessions: async_sessionmaker[AsyncSession],
    material_id: int,
    adapter: SourceAdapter,
    source_path: str,
    settings: Settings,
) -> CostEstimate:
    """Измеряет источник, сохраняет оценку и при нужде останавливает конвейер.

    Поднимает `AwaitingUser`, если предварительная оценка превышает
    `COST_WARN_THRESHOLD`. Исключением, а не возвращаемым значением: §4.5
    требует, чтобы задача завершилась, а забытая проверка возвращаемого
    значения дала бы молчаливое продолжение конвейера, то есть обход гейта.
    """
    meta = await adapter.probe(source_path)
    estimate = estimate_from_probe(meta, settings)

    # Исключение поднимается ПОСЛЕ закрытия транзакции, а не внутри.
    # Первая версия поднимала его внутри `session_scope`, и откат отменял саму
    # постановку на ожидание: материал не вставал в `awaiting_user` вовсе, а
    # задача завершалась как будто по §4.5. Дефект нашёл
    # `test_gate_parks_material_with_a_question`.
    question: str | None = None

    async with session_scope(sessions) as session:
        await _save_estimate(session, material_id, estimate)
        if exceeds_threshold(estimate, settings):
            question = gate_question(estimate, meta.pages)
            await park(session, material_id, question, kind=GATE_KIND)

    if question is not None:
        log.info("budget_gate_fired", material_id=material_id, cost=str(estimate.est_cost))
        raise AwaitingUser(question)

    return estimate


async def reestimate_after_extract(
    sessions: async_sessionmaker[AsyncSession],
    material_id: int,
    preliminary: CostEstimate,
    settings: Settings,
) -> CostEstimate:
    """Пересчитывает оценку по фактическим фрагментам и спрашивает повторно.

    §6.3: «Если уточнённая оценка после `extract` превышает предварительную
    более чем в `COST_REESTIMATE_FACTOR` раз, пользователь спрашивается
    повторно тем же механизмом. Без этой проверки гейт обходится любым
    материалом, чью стоимость `probe` недооценил».

    Работает и при выключенном абсолютном гейте: вопрос задаётся по
    относительному росту, а не по сумме.
    """
    # Исключение — после коммита, по той же причине, что в `probe_and_gate`.
    question: str | None = None

    async with session_scope(sessions) as session:
        fragments, tokens = (
            await session.execute(
                select(func.count(), func.coalesce(func.sum(Fragment.tokens), 0)).where(
                    Fragment.material_id == material_id
                )
            )
        ).one()

        refined = estimate_after_extract(
            fragments=int(fragments or 0),
            fragment_tokens=int(tokens or 0),
            settings=settings,
        )
        await _save_estimate(session, material_id, refined)

        if needs_reestimate_confirmation(preliminary, refined, settings):
            question = (
                "Материал оказался больше, чем показала первая оценка: "
                f"≈ {refined.est_cost:.2f} {refined.currency} вместо "
                f"≈ {preliminary.est_cost:.2f}."
            )
            await park(session, material_id, question, kind=GATE_KIND)

    if question is not None:
        log.info(
            "budget_reestimate_fired",
            material_id=material_id,
            preliminary=str(preliminary.est_cost),
            refined=str(refined.est_cost),
        )
        raise AwaitingUser(question)

    return refined


async def _save_estimate(session: AsyncSession, material_id: int, estimate: CostEstimate) -> None:
    """Пишет оценку в `materials.cost_estimate` (§3).

    Перезаписывает, а не дополняет: §6.3 говорит «уточнённая оценка
    перезаписывается там же». Хранить обе значило бы вводить поле, которого
    в §3 нет.
    """
    await session.execute(
        update(Material).where(Material.id == material_id).values(cost_estimate=estimate.as_json())
    )


async def choose_notes_only(session: AsyncSession, material_id: int) -> None:
    """Кнопка «Только конспект, без тестов» (§6.3).

    Переводит материал в `notes_only`. Стадия `questions` будет пропущена
    конвейером с отметкой `questions:skipped`, а `schedule` выполнится и не
    создаст ни одной строки `review_states`: у фактов нет валидных вопросов,
    значит они не обучаемы (§14.3). Отдельного условия для этого не нужно —
    работает предикат.
    """
    await session.execute(
        update(Material).where(Material.id == material_id).values(mode="notes_only")
    )
    log.info("material_notes_only", material_id=material_id)


async def choose_full(session: AsyncSession, material_id: int) -> None:
    """Кнопка «Обработать полностью» (§6.3) и «Достроить задания».

    Вторая вдобавок снимает отметки `questions:skipped` и `schedule`, чтобы
    обычное возобновление §4.3 начало с `questions` — см.
    `core.ingest.stages.reopen_for_questions`.
    """
    await session.execute(update(Material).where(Material.id == material_id).values(mode="full"))
    log.info("material_full_mode", material_id=material_id)
