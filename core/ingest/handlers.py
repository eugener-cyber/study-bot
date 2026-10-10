"""Обработчики стадий.

Шесть стадий §4.2. `extract` работает через адаптер (§5.1), `schedule`
создаёт расписание (§4.6), три содержательные — `sections`, `facts`,
`questions` — вызывают модель через клиент, а `notes` собирает конспект из
сохранённых фактов без вызова (§8, разбор в докстроке обработчика).

До WP-04 четыре содержательные стадии были заглушками, и критерий приёмки
WP-03 это допускал: «`make seed` доводит материал до `ready` на заглушках».
Суффикс `_stub` ушёл вместе с ними.

Каждый обработчик соблюдает внутренний порядок §4.2: подготовить, проверить,
удалить предыдущие частичные артефакты своей стадии, сохранить результат.
Отметку в `completed_stages` ставит конвейер.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.adapters.base import IngestCtx
from core.config import Settings
from core.db.models import Fact, Fragment, Question, ReviewState, Section
from core.db.teachable import generatable_facts_select, teachable_facts_select
from core.ingest.pipeline import StageContext, StageHandler
from core.llm.client import LLMClient
from core.llm.prompts import load_prompt
from core.llm.schemas import (
    FactsAnswer,
    QuestionDraft,
    QuestionSet,
    SectionPlan,
    SectionsAnswer,
)
from core.logging import get_logger

log = get_logger(__name__)


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


# ------------------------------------------------------------ содержательные
#
# Четыре стадии §4.2, которым нужна модель. До WP-04 здесь стояли заглушки;
# теперь — вызовы через клиент. §2.1 требует, чтобы стадия видела только его:
# ни провайдера, ни режимов, ни токенов здесь нет, и это проверяется тестом на
# импорты, а не соглашением.
#
# **Границы пакета, названные прямо.** Промпты `sections_v1`, `facts_v1`,
# `questions_v1` приезжают вместе с этим пакетом, хотя их содержание
# принадлежит WP-07 и WP-09. Иначе конвейер не выполняется вовсе, и критерий
# приёмки самого WP-04 — «все вызовы попадают в `llm_calls`» на настоящих
# стадиях — проверить нечем. Версия в имени файла (§6.1) для этого и
# существует: WP-07 и WP-09 выпускают `_v2`, и старые записи становятся
# недействительными автоматически.
#
# Сознательно **не** реализовано здесь, с указанием владельца:
# §6.2 нарезка секций по `MAX_SECTION_TOKENS` — WP-07; §7.4 дедуп фактов —
# WP-07; §8.1 выдача конспекта с пагинацией — WP-08; §10.2 повторная
# генерация набора — WP-09; §11 валидация вопросов — WP-10.


def _fragment_listing(rows: Sequence[tuple[int, str]]) -> str:
    """Фрагменты для промпта: идентификатор в начале строки.

    Идентификатор подаётся модели потому, что §7.3 проверка 1 требует ссылок
    на фрагменты, а сослаться можно только на то, что видно. Формат `[12]` —
    самый дешёвый для разбора глазами при отладке промпта.
    """
    return "\n".join(f"[{fragment_id}] {text or ''}".strip() for fragment_id, text in rows)


async def _material_fragments(session: AsyncSession, material_id: int) -> list[tuple[int, str]]:
    rows = await session.execute(
        select(Fragment.id, Fragment.text)
        .where(Fragment.material_id == material_id)
        .order_by(Fragment.ord)
    )
    return [(int(fragment_id), str(text or "")) for fragment_id, text in rows.all()]


async def _fragments_by_ids(
    session: AsyncSession, fragment_ids: Sequence[int]
) -> list[tuple[int, str]]:
    if not fragment_ids:
        return []
    rows = await session.execute(
        select(Fragment.id, Fragment.text)
        .where(Fragment.id.in_(list(fragment_ids)))
        .order_by(Fragment.ord)
    )
    return [(int(fragment_id), str(text or "")) for fragment_id, text in rows.all()]


UNASSIGNED_TITLE = "Не отнесено к темам"
"""Секция для фрагментов, которые модель не распределила.

Промпт требует, чтобы ни один фрагмент не потерялся, но полагаться на это
нельзя. Из двух реакций на потерю — уронить материал или собрать остаток в
отдельную секцию — выбрана вторая: упавший материал виден, но пользователь
теряет всё, а остаток в своей секции сохраняет содержание и видно в логе по
событию `sections_unassigned`. Молчаливой потери не остаётся ни в одном из
вариантов, и это единственное, что было обязательным.
"""


def make_sections_handler(client: LLMClient) -> StageHandler:
    """Стадия `sections`: группировка фрагментов в смысловые части (§7.1).

    Один вызов на материал. Нарезка по `MAX_SECTION_TOKENS` (§6.2) появится в
    WP-07; до неё материал, не влезающий в контекст, упадёт на стороне
    провайдера с внятной ошибкой, а не будет молча обработан частично.
    """

    async def handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        rows = await _material_fragments(session, ctx.material_id)
        if not rows:
            raise ValueError("нет фрагментов — стадия extract не выполнена")

        prompt = load_prompt("sections")
        answer = (
            await client.structured(
                prompt.render(fragments=_fragment_listing(rows)),
                SectionsAnswer,
                purpose="sections",
                prompt_version=prompt.version,
                user_id=_user_id(material),
                material_id=ctx.material_id,
            )
        ).value

        known = [fragment_id for fragment_id, _ in rows]
        plans = _assign_fragments(answer.sections, known, material_id=ctx.material_id)

        await session.execute(delete(Section).where(Section.material_id == ctx.material_id))
        session.add_all(
            [
                Section(
                    material_id=ctx.material_id,
                    ord=position,
                    title=title,
                    summary_md="",
                    fragment_ids=fragment_ids,
                )
                for position, (title, fragment_ids) in enumerate(plans, start=1)
            ]
        )
        await session.flush()
        log.info("sections_built", material_id=ctx.material_id, count=len(plans))

    return handler


def _assign_fragments(
    plans: Sequence[SectionPlan], known: Sequence[int], *, material_id: int
) -> list[tuple[str, list[int]]]:
    """Раскладывает фрагменты по секциям ответа модели.

    Три защиты, и каждая закрывает свой способ испортить материал.
    Выдуманный идентификатор отбрасывается — §7.3 проверка 1 по смыслу: без
    этого ссылки фиктивны. Фрагмент, попавший в две секции, остаётся в первой
    — иначе один и тот же текст породит два набора фактов, и дедуп §7.4
    получит работу, которой могло не быть. Нераспределённый остаток уходит в
    отдельную секцию.
    """
    if not known:
        raise ValueError("нечего распределять: список фрагментов пуст")

    available = set(known)
    taken: set[int] = set()
    result: list[tuple[str, list[int]]] = []

    for plan in plans:
        fragment_ids = [
            fragment_id
            for fragment_id in plan.fragment_ids
            if fragment_id in available and fragment_id not in taken
        ]
        unknown = [fragment_id for fragment_id in plan.fragment_ids if fragment_id not in available]
        if unknown:
            log.warning(
                "sections_unknown_fragments",
                material_id=material_id,
                title=plan.title,
                unknown=unknown,
            )
        if not fragment_ids:
            continue
        taken.update(fragment_ids)
        result.append((plan.title, fragment_ids))

    leftover = [fragment_id for fragment_id in known if fragment_id not in taken]
    if leftover:
        log.warning("sections_unassigned", material_id=material_id, count=len(leftover))
        result.append((UNASSIGNED_TITLE, leftover))

    # Проверки «ни одной секции не вышло» здесь нет намеренно, и это не
    # недосмотр: при непустом `known` остаток гарантирует хотя бы одну
    # секцию, то есть такая проверка была бы недостижимым кодом. Худший
    # случай — ответ целиком из выдуманных идентификаторов, и тогда материал
    # становится одной секцией `UNASSIGNED_TITLE`: содержание сохранено,
    # факты по нему извлекутся, а перекос виден в журнале по двум
    # предупреждениям подряд. Падение здесь стоило бы дороже — пользователь
    # потерял бы материал из-за ошибки модели в нумерации.
    return result


def norm_hash(statement: str) -> str:
    """`facts.norm_hash` по §7.4: NFKC, lowercase, без пунктуации, один пробел.

    Полный дедуп §7.4 — сравнение кандидатов `rapidfuzz` и разрешение
    конфликтов по богатству evidence — принадлежит WP-07. Здесь только
    первичный признак, потому что колонка §3 обязательна и заполнить её
    случайным хешем значило бы сделать будущий дедуп невозможным: кандидаты
    находятся сравнением именно этого поля.
    """
    normalized = unicodedata.normalize("NFKC", statement).lower()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in normalized
    )
    return hashlib.sha256(" ".join(normalized.split()).encode()).hexdigest()


def make_facts_handler(client: LLMClient, settings: Settings) -> StageHandler:
    """Стадия `facts`: извлечение фактов по секциям (§7.1, §7.2, §7.3).

    Один вызов на секцию, как требует §7.1. Ответ приносит и `summary_md`
    секции — именно поэтому стадия конспекта модель не вызывает: §8 строит
    конспект из фактов и этого поля.
    """

    async def handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        sections = (
            await session.execute(
                select(Section.id, Section.title, Section.fragment_ids)
                .where(Section.material_id == ctx.material_id)
                .order_by(Section.ord)
            )
        ).all()
        if not sections:
            raise ValueError("нет секций — стадия sections не выполнена")

        prompt = load_prompt("facts")
        await session.execute(delete(Fact).where(Fact.material_id == ctx.material_id))

        stored = 0
        derived = 0
        for section_id, title, fragment_ids in sections:
            rows = await _fragments_by_ids(session, list(fragment_ids or []))
            if not rows:
                log.warning("facts_section_without_fragments", section_id=int(section_id))
                continue

            answer = (
                await client.structured(
                    prompt.render(title=str(title), fragments=_fragment_listing(rows)),
                    FactsAnswer,
                    purpose="facts",
                    prompt_version=prompt.version,
                    user_id=_user_id(material),
                    material_id=ctx.material_id,
                )
            ).value

            await session.execute(
                update(Section)
                .where(Section.id == int(section_id))
                .values(title=answer.title, summary_md=answer.summary_md)
            )

            submitted = {fragment_id for fragment_id, _ in rows}
            for draft in answer.facts:
                evidence = [
                    fragment_id for fragment_id in draft.fragment_ids if fragment_id in submitted
                ]
                if not evidence:
                    # §7.3 проверка 1: `fragment_ids ⊆ submitted_fragment_ids`,
                    # нарушение — факт отбрасывается. Это защита И-1: без неё
                    # вся цепочка ссылок фиктивна.
                    log.warning(
                        "fact_dropped_without_evidence",
                        material_id=ctx.material_id,
                        section_id=int(section_id),
                        claimed=draft.fragment_ids,
                    )
                    continue

                session.add(
                    Fact(
                        material_id=ctx.material_id,
                        section_id=int(section_id),
                        statement=draft.statement,
                        detail=draft.detail,
                        kind=draft.kind,
                        fragment_ids=evidence,
                        importance=draft.importance,
                        difficulty=draft.difficulty,
                        confidence=draft.confidence,
                        derived=draft.derived,
                        generation_status="pending",
                        norm_hash=norm_hash(draft.statement),
                        created_at=datetime.now(UTC),
                    )
                )
                stored += 1
                derived += int(draft.derived)

        await session.flush()
        if not stored:
            raise ValueError("ни одного проверяемого утверждения: в материале нечему учиться")

        ratio = derived / stored
        log.info(
            "facts_extracted",
            material_id=ctx.material_id,
            count=stored,
            derived=derived,
            derived_ratio=round(ratio, 3),
        )
        if ratio > settings.DERIVED_WARN_RATIO:
            # §7.3, наблюдаемость проверки 4: доля выведенных фактов выше
            # порога означает, что значительная часть знания не попадёт в
            # тесты. Сообщение пользователю §7.3 требует отдельно, и его
            # владелец — пакет, который отвечает за показ конспекта; здесь
            # событие в журнале, чтобы перекос был виден уже сейчас, а не
            # обнаружился по подозрительно малому числу вопросов.
            log.warning(
                "derived_ratio_above_threshold",
                material_id=ctx.material_id,
                ratio=round(ratio, 3),
                threshold=settings.DERIVED_WARN_RATIO,
            )

    return handler


WARNING_MARK = "⚠️"
"""Пометка §8 у выведенных фактов и фактов с низкой уверенностью."""


def make_notes_handler(settings: Settings) -> StageHandler:
    """Стадия `notes`: конспект **собирается**, а не генерируется (§8).

    Модель здесь не вызывается, и это прочтение самого §8, а не экономия.
    Шаблон §8 ссылается на `section.summary_md` как на **вход**, а четыре его
    блока выводит из сохранённых фактов: «Ключевые понятия» — факты вида
    `definition`, «Основные положения» — факты секции, «Связи и отличия» —
    `difference` и `cause`, «Что запомнить» — факты с `importance = 3`.
    Отдельной таблицы под конспект §3 не содержит, а §8.1 отдаёт его с
    пагинацией по требованию. Значит хранить нечего: конспект — функция от
    фактов, и любая его копия в базе разошлась бы с ними после первой
    перегенерации.

    Стадия поэтому ничего не записывает, но и пустой не является: она
    собирает документ и падает, если собрать нечего. Без этой проверки
    материал доходил бы до `ready` с конспектом, который при первом открытии
    окажется пустым экраном.

    Расхождение с §6.3, где `notes` учитывается как стадия с расходом
    токенов, вынесено change request'ом — решать его за Архитектора §40
    запрещает. До решения оценка остаётся завышенной, что безопаснее
    заниженной.
    """

    async def handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        document = await render_note(session, ctx.material_id, settings)
        if not document.strip():
            raise ValueError("конспект собрался пустым: нет ни секций, ни фактов")
        log.info(
            "note_rendered",
            material_id=ctx.material_id,
            chars=len(document),
        )

    return handler


@dataclass(frozen=True)
class _NoteFact:
    """Факт в том виде, в каком его читает шаблон §8.

    Строки запроса превращаются в объект сразу, а не разбираются индексами по
    месту: `fact[3] == "definition"` в четырёх разных выборках шаблона — это
    четыре возможности перепутать колонку, и опечатка дала бы не ошибку, а
    тихо неверный раздел конспекта.
    """

    fact_id: int
    section_id: int
    statement: str
    kind: str
    importance: int
    flagged: bool
    """`⚠️` по §8: выведенный факт или факт с низкой `final_confidence`."""


async def render_note(session: AsyncSession, material_id: int, settings: Settings) -> str:
    """Собирает конспект материала по шаблону §8.

    Функция лежит здесь временно: её место — пакет выдачи конспекта (WP-08),
    где появятся три режима глубины, пагинация §8.1 и выгрузка. Разделение
    фактов на чистые и помеченные берётся из того же предиката §7.3, что
    использует генерация вопросов, — иначе `⚠️` в конспекте и отсутствие
    вопросов перестали бы совпадать, и пользователь видел бы факт без
    пометки, по которому его никогда не спросят.
    """
    sections = (
        await session.execute(
            select(Section.id, Section.title, Section.summary_md)
            .where(Section.material_id == material_id)
            .order_by(Section.ord)
        )
    ).all()

    generatable = generatable_facts_select(settings.MIN_FACT_CONFIDENCE).subquery()
    clean = {
        int(row[0])
        for row in (
            await session.execute(
                select(generatable.c.fact_id).where(generatable.c.material_id == material_id)
            )
        ).all()
    }

    facts = [
        _NoteFact(
            fact_id=int(row[0]),
            section_id=int(row[1] or 0),
            statement=str(row[2]),
            kind=str(row[3] or ""),
            importance=int(row[4] or 0),
            flagged=int(row[0]) not in clean,
        )
        for row in (
            await session.execute(
                select(Fact.id, Fact.section_id, Fact.statement, Fact.kind, Fact.importance)
                .where(Fact.material_id == material_id)
                .order_by(Fact.id)
            )
        ).all()
    ]

    parts: list[str] = []
    for section_id, title, summary_md in sections:
        own = [fact for fact in facts if fact.section_id == int(section_id)]
        summary = str(summary_md or "")
        if not own and not summary.strip():
            continue

        parts.append(f"## {title}\n{summary}".rstrip())
        for heading, selected in (
            ("Ключевые понятия", [f for f in own if f.kind == "definition"]),
            ("Основные положения", own),
            ("Связи и отличия", [f for f in own if f.kind in {"difference", "cause"}]),
            ("Что запомнить", [f for f in own if f.importance == 3]),
        ):
            if not selected:
                continue
            body = "\n".join(_note_line(fact) for fact in selected)
            parts.append(f"### {heading}\n{body}")

    return "\n\n".join(parts)


def _note_line(fact: _NoteFact) -> str:
    mark = f" {WARNING_MARK}" if fact.flagged else ""
    return f"— {fact.statement}{mark}"


def make_questions_handler(client: LLMClient, settings: Settings) -> StageHandler:
    """Стадия `questions`: один вызов на факт (§10.1).

    Факты отбираются предикатом §7.3 — не обучаемости: обучаемость требует
    существующего валидного вопроса, и на этой стадии она пуста по построению.

    Статус вопроса — `valid`, как и в WP-03. Правильный статус `draft` до
    валидатора §11.5, но валидатор приезжает в WP-10, и `draft` означал бы,
    что до WP-10 продукт не задаёт ни одного вопроса: §13.3 отбирает только
    `valid`. Это временно и названо здесь, а не оставлено на догадку.

    **Удаление прежних `valid` перед вставкой уйдёт вместе со статусом**
    (Issue #28). Сегодня оно необходимо: §3.1 держит частичный уникальный
    индекс по слоту, и повторный прогон иначе упёрся бы в него, потеряв с
    транзакцией и уже сгенерированные наборы. Но §4.2 называет частичным
    артефактом именно `draft`, а `review_logs.question_id` и
    `session_items.question_id` объявлены `ForeignKey` без `ondelete`: как
    только по вопросам начнут отвечать (WP-05), удаление отвеченного вопроса
    поднимет нарушение внешнего ключа, а §38 №4 запрещает ломать историю
    факта. Сегодня путь недостижим — отвечать ещё некому, а
    `reopen_for_questions` ведёт от `notes_only`, где вопросов не было.
    """

    async def handler(session: AsyncSession, material: object, ctx: StageContext) -> None:
        generatable = generatable_facts_select(settings.MIN_FACT_CONFIDENCE).subquery()
        fact_ids = [
            int(row[0])
            for row in (
                await session.execute(
                    select(generatable.c.fact_id)
                    .where(generatable.c.material_id == ctx.material_id)
                    .order_by(generatable.c.fact_id)
                )
            ).all()
        ]
        if not fact_ids:
            raise ValueError(
                "нет фактов, по которым можно задать вопрос: все ниже порога "
                "MIN_FACT_CONFIDENCE или помечены как выводы модели"
            )

        await _drop_drafts(session, ctx.material_id)
        prompt = load_prompt("questions")
        total = 0

        for fact_id in fact_ids:
            statement, detail, fragment_ids = (
                await session.execute(
                    select(Fact.statement, Fact.detail, Fact.fragment_ids).where(Fact.id == fact_id)
                )
            ).one()
            rows = await _fragments_by_ids(session, list(fragment_ids or []))

            result = await client.structured(
                prompt.render(
                    statement=str(statement),
                    detail=str(detail),
                    fragments=_fragment_listing(rows),
                ),
                QuestionSet,
                purpose="questions",
                prompt_version=prompt.version,
                user_id=_user_id(material),
                material_id=ctx.material_id,
            )
            answer = result.value

            await session.execute(
                delete(Question).where(Question.fact_id == fact_id, Question.status == "valid")
            )
            for question in _unique_slots(answer.questions, fact_id=fact_id):
                session.add(
                    Question(
                        fact_id=fact_id,
                        type=question.type,
                        direction=question.direction,
                        payload=question.payload.model_dump(mode="json"),
                        answer=question.answer.model_dump(mode="json"),
                        difficulty=question.difficulty,
                        status="valid",
                        prompt_version=prompt.version,
                        model=result.model,
                        provider=client.provider_name,
                        generation_attempt=1,
                        shown_count=0,
                        created_at=datetime.now(UTC),
                    )
                )
                total += 1

            await session.execute(
                update(Fact).where(Fact.id == fact_id).values(generation_status="ok")
            )

        await session.flush()
        log.info(
            "questions_generated",
            material_id=ctx.material_id,
            facts=len(fact_ids),
            questions=total,
        )

    return handler


def _unique_slots(questions: Sequence[QuestionDraft], *, fact_id: int) -> list[QuestionDraft]:
    """Оставляет по одному вопросу на пару `(type, direction)`.

    Не проверка качества, а условие записи: §3.1 держит частичный уникальный
    индекс `uq_questions_fact_type_direction_valid`, и второй вопрос в том же
    слоте уронил бы вставку `IntegrityError`, потеряв заодно и первый —
    транзакция стадии откатилась бы целиком.

    Остальные правила §10.2 — минимум два разных типа, наличие `open` по
    CR-1б, повторная генерация набора — здесь не применяются: §10.2
    предписывает им **перегенерацию** с переходом в `single_format` после
    второй неудачи, и это работа WP-09.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[QuestionDraft] = []
    for question in questions:
        slot = (question.type, question.direction)
        if slot in seen:
            log.info("question_slot_duplicate", fact_id=fact_id, slot=f"{slot[0]}:{slot[1]}")
            continue
        seen.add(slot)
        kept.append(question)
    return kept


def _user_id(material: object) -> int | None:
    """`user_id` материала для строки `llm_calls` (§3).

    Через `getattr`, потому что протокол `StageHandler` объявляет `material`
    как объект, а тесты конвейера передают заготовку. Поле необязательно: §3
    допускает `NULL`, и вызов без владельца лучше, чем падение учёта.
    """
    value = getattr(material, "user_id", None)
    return int(value) if isinstance(value, int) else None


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


def build_handlers(client: LLMClient, settings: Settings) -> dict[str, StageHandler]:
    """Карта стадий §4.2 — одна на весь проект.

    Воркер и харнесс `make seed` собирали её каждый сам, и до WP-04 это
    стоило дёшево: обе карты ссылались на заглушки без зависимостей. С
    появлением клиента цена выросла — забытая стадия в одной из карт даёт
    материал, который доходит до `ready`, пропустив вызов модели, и выглядит
    это как плохо работающая модель, а не как пропущенная стадия.

    Имена стадий не перечисляются литералами дважды: состав проверяется
    против `core.ingest.stages.ORDER` тестом.
    """
    return {
        "extract": extract,
        "sections": make_sections_handler(client),
        "facts": make_facts_handler(client, settings),
        "notes": make_notes_handler(settings),
        "questions": make_questions_handler(client, settings),
        "schedule": make_schedule_handler(settings),
    }


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
