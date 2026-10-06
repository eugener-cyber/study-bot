"""Обработчики стадий.

Настоящих здесь две: `extract` (§5.1, работает через адаптер) и `schedule`
(§4.6). Четыре содержательные — `sections`, `facts`, `notes`, `questions` —
заглушки до WP-04, где появится LLM-слой. Так и записано в плане работ:
критерий приёмки WP-03 звучит «`make seed` доводит материал до `ready` на
заглушках».

Заглушки дают **правдоподобные** данные, а не пустые: факт со ссылкой на
фрагмент, вопрос со статусом `valid`. Иначе стадия `schedule` не создала бы ни
одной строки `review_states` (нет обучаемых фактов по §14.3), и ни механизм
§4.6, ни сквозной скелет WP-05 проверить было бы нечем.

Каждый обработчик соблюдает внутренний порядок §4.2: подготовить, проверить,
удалить предыдущие частичные артефакты своей стадии, сохранить результат.
Отметку в `completed_stages` ставит конвейер.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.adapters.base import IngestCtx
from core.config import Settings
from core.db.models import Fact, Fragment, Question, ReviewState, Section
from core.db.teachable import teachable_facts_select
from core.ingest.pipeline import StageContext, StageHandler
from core.logging import get_logger

log = get_logger(__name__)

STUB_PROMPT_VERSION = "stub-0"
"""Версия «промпта» заглушки.

Записывается в `questions.prompt_version` по §3. Отдельное значение, а не
пустая строка: §11 обещает, что смена версии промпта делает старые записи
недействительными, и вопросы, сгенерированные заглушкой, должны отличаться от
настоящих по этому полю, когда появится WP-04.
"""


async def extract(session: AsyncSession, material: object, ctx: StageContext) -> None:
    """Разбирает источник на фрагменты через адаптер (§5.1).

    Частичные артефакты стадии — строки `fragments` материала; при повторном
    запуске они удаляются целиком. Выборочное обновление здесь невозможно:
    порядковые номера фрагментов задаёт адаптер, и при изменении источника
    старый фрагмент №3 и новый №3 — разные тексты с одним номером.
    """
    stream = ctx.adapter.extract(ctx.source_path, _ingest_ctx(ctx))
    collected = [fragment async for fragment in stream]
    if not collected:
        raise ValueError("адаптер не дал ни одного фрагмента")
    for fragment in collected:
        if not fragment.locator:
            raise ValueError(f"фрагмент {fragment.ord} без локатора — нарушение §3")

    await session.execute(delete(Fragment).where(Fragment.material_id == ctx.material_id))
    session.add_all(
        [
            Fragment(
                material_id=ctx.material_id,
                ord=item.ord,
                kind=item.kind,
                text=item.text,
                locator=item.locator,
                quality=item.quality,
                tokens=item.tokens,
            )
            for item in collected
        ]
    )
    await session.flush()
    log.info("fragments_extracted", material_id=ctx.material_id, count=len(collected))


def _ingest_ctx(ctx: StageContext) -> IngestCtx:
    """Контекст для адаптера: он не должен знать про стадии и сессии."""
    return IngestCtx(material_id=ctx.material_id, tmp_dir=str(ctx.tmp_dir))


async def sections_stub(session: AsyncSession, material: object, ctx: StageContext) -> None:
    """Заглушка: одна секция на весь материал.

    Настоящая нарезка — WP-07, по §6.2 с `MAX_SECTION_TOKENS`. Одна секция
    достаточна для проверки конвейера и честна: она не делает вид, что
    материал разобран на темы.
    """
    fragment_ids = list(
        (
            await session.execute(
                select(Fragment.id)
                .where(Fragment.material_id == ctx.material_id)
                .order_by(Fragment.ord)
            )
        ).scalars()
    )
    if not fragment_ids:
        raise ValueError("нет фрагментов — стадия extract не выполнена")

    await session.execute(delete(Section).where(Section.material_id == ctx.material_id))
    session.add(
        Section(
            material_id=ctx.material_id,
            ord=1,
            title="Материал целиком (заглушка)",
            summary_md="",
            fragment_ids=fragment_ids,
        )
    )
    await session.flush()


async def facts_stub(session: AsyncSession, material: object, ctx: StageContext) -> None:
    """Заглушка: один факт на фрагмент, evidence — сам фрагмент.

    `confidence` берётся высокой, чтобы факт прошёл порог §7.3 и дошёл до
    `schedule`: иначе непроверяемой осталась бы вся цепочка от обучаемости до
    создания `review_states`. `derived` остаётся `false` — заглушка не
    выдумывает выводов.
    """
    section_id = (
        (
            await session.execute(
                select(Section.id)
                .where(Section.material_id == ctx.material_id)
                .order_by(Section.ord)
            )
        )
        .scalars()
        .first()
    )
    if section_id is None:
        raise ValueError("нет секций — стадия sections не выполнена")

    fragments = list(
        (
            await session.execute(
                select(Fragment.id, Fragment.text)
                .where(Fragment.material_id == ctx.material_id)
                .order_by(Fragment.ord)
            )
        ).all()
    )

    await session.execute(delete(Fact).where(Fact.material_id == ctx.material_id))
    for fragment_id, text in fragments:
        statement = (text or "Фрагмент без текста")[:200]
        session.add(
            Fact(
                material_id=ctx.material_id,
                section_id=section_id,
                statement=statement,
                detail=statement,
                kind="definition",
                fragment_ids=[fragment_id],
                importance=2,
                difficulty=2,
                confidence=0.9,
                derived=False,
                generation_status="pending",
                norm_hash=hashlib.sha256(statement.encode()).hexdigest(),
                created_at=datetime.now(UTC),
            )
        )
    await session.flush()


async def notes_stub(session: AsyncSession, material: object, ctx: StageContext) -> None:
    """Заглушка конспекта: перечисление утверждений фактов.

    Настоящий конспект — WP-08, по структуре §8 с четырьмя разделами. Здесь
    заполняется `sections.summary_md`, чтобы §8.1 было что отдавать и чтобы
    пустое поле не выглядело сбоем стадии.
    """
    statements = list(
        (
            await session.execute(
                select(Fact.statement).where(Fact.material_id == ctx.material_id).order_by(Fact.id)
            )
        ).scalars()
    )
    if not statements:
        raise ValueError("нет фактов — стадия facts не выполнена")

    body = "\n".join(f"- {s}" for s in statements)
    await session.execute(
        update(Section)
        .where(Section.material_id == ctx.material_id)
        .values(summary_md=f"## Основные положения (заглушка)\n\n{body}\n")
    )


async def questions_stub(session: AsyncSession, material: object, ctx: StageContext) -> None:
    """Заглушка: по два вопроса на факт — `mcq` и `open`.

    Два разных типа, как требует §10.2, и один из них порождающий, как требует
    §10.1 после CR-1б. Это не случайно: с одним только `mcq` факт попал бы под
    потолок `RECOGNITION_ONLY_CAP` (§14.1), и mastery в сквозном прогоне
    вела бы себя не так, как в работе.

    **Частичный артефакт этой стадии — строки со `status = 'draft'`** (§4.2):
    вопрос получает окончательный статус только после валидатора (§11.5), то
    есть после сетевого вызова. Перезапуск удаляет все `draft` материала до
    начала работы. Без этого упавший воркер оставляет занятые слоты
    `(fact_id, type, direction)` — §38 №20.
    """
    fact_ids = list(
        (
            await session.execute(
                select(Fact.id).where(Fact.material_id == ctx.material_id).order_by(Fact.id)
            )
        ).scalars()
    )
    if not fact_ids:
        raise ValueError("нет фактов — стадия facts не выполнена")

    await _drop_drafts(session, ctx.material_id)

    for fact_id in fact_ids:
        for question_type, payload in (
            ("mcq", {"stem": "Что верно?", "options": ["а", "б", "в", "г"]}),
            ("open", {"stem": "Сформулируйте утверждение своими словами."}),
        ):
            session.add(
                Question(
                    fact_id=fact_id,
                    type=question_type,
                    direction="forward",
                    payload=payload,
                    answer={"correct": 0} if question_type == "mcq" else {"text": "заглушка"},
                    difficulty=2,
                    status="valid",
                    prompt_version=STUB_PROMPT_VERSION,
                    model="stub",
                    provider="manual",
                    generation_attempt=1,
                    shown_count=0,
                    created_at=datetime.now(UTC),
                )
            )
    await session.flush()


async def _drop_drafts(session: AsyncSession, material_id: int) -> None:
    """Удаляет `draft`-строки материала — §4.2 для стадии `questions`.

    Отдельной функцией, потому что вызывается и из перезапуска стадии, и из
    регенерации (§12.2, WP-10). Два экземпляра этого удаления разошлись бы, а
    цена расхождения — занятый навсегда слот генерации.
    """
    fact_ids = select(Fact.id).where(Fact.material_id == material_id).scalar_subquery()
    await session.execute(
        delete(Question).where(Question.fact_id.in_(fact_ids), Question.status == "draft")
    )


def make_schedule_handler(settings: Settings) -> StageHandler:
    """Фабрика обработчика `schedule`: ему нужны настройки, остальным — нет.

    Фабрика, а не параметр в протоколе `StageHandler`: иначе все шесть
    обработчиков получали бы настройки, из которых пятерым нужен только порог
    обучаемости, и зависимость расползлась бы по стадиям.
    """

    async def handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        await ensure_review_states(session, ctx.material_id, settings)

    return handler


async def ensure_review_states(session: AsyncSession, material_id: int, settings: Settings) -> int:
    """Идемпотентно создаёт `review_states` для обучаемых фактов. §4.6.

    Возвращает число вставленных строк.

    Вставка идемпотентна (`ON CONFLICT DO NOTHING`): повторный запуск стадии
    не сбрасывает накопленное состояние повторения. Прямое следствие И-7 —
    идентичность факта стабильна, и его расписание не обнуляется при
    переобработке материала (§38 №16).

    Факты, не прошедшие фильтр обучаемости, строку не получают вовсе. Предикат
    §14.3 один и тот же для трёх потребителей, и здесь он вызывается через ту
    же обёртку, что в статистике и планировщике.

    §4.6 требует вызова из двух мест: стадии `schedule` и успешного завершения
    регенерации (§12.2). Второй появится в WP-10, и функция написана так,
    чтобы он не потребовал правок: §38 №19 обещает, что факт «возвращается в
    ротацию автоматически», и без явного вызова обещание не выполняется.

    `due_at` считается от текущего момента, а не от `materials.processed_at`:
    §4.6 пишет `processed_at + NEW_FACT_DELAY`, но `processed_at` проставляется
    при переходе материала в `ready`, то есть после этой стадии, и на момент
    вставки ещё пуст. Текущий момент и есть момент обработки.
    """
    # Фильтр по материалу идёт прямо по колонке предиката: `teachable_facts`
    # возвращает `material_id` — именно для того, чтобы потребителям не
    # приходилось присоединять `facts` второй раз.
    teachable = teachable_facts_select(settings.MIN_FACT_CONFIDENCE).subquery()
    result = await session.execute(
        select(teachable.c.fact_id).where(teachable.c.material_id == material_id)
    )
    rows: list[int] = [int(row[0]) for row in result.all()]
    if not rows:
        log.info("no_teachable_facts", material_id=material_id)
        return 0

    user_id = await _owner_of_material(session, material_id)

    due_at = datetime.now(UTC) + timedelta(minutes=settings.NEW_FACT_DELAY_MIN)
    statement = (
        pg_insert(ReviewState)
        .values(
            [
                {
                    "user_id": user_id,
                    "fact_id": fact_id,
                    "due_at": due_at,
                    "state": 0,
                    "reps": 0,
                    "lapses": 0,
                    "seen": False,
                    "suspended": False,
                    "created_at": func.now(),
                }
                for fact_id in rows
            ]
        )
        .on_conflict_do_nothing(index_elements=["user_id", "fact_id"])
        .returning(ReviewState.fact_id)
    )
    inserted = list((await session.execute(statement)).scalars())
    log.info(
        "review_states_ensured",
        material_id=material_id,
        teachable=len(rows),
        inserted=len(inserted),
    )
    return len(inserted)


async def _owner_of_material(session: AsyncSession, material_id: int) -> int:
    from core.db.models import Material

    return int(
        (
            await session.execute(select(Material.user_id).where(Material.id == material_id))
        ).scalar_one()
    )
