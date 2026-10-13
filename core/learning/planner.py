"""Сбор сессии: отбор кандидатов по сроку. ТЗ §16.

**Планировщик минимален, и минимальность здесь названа.** §16 описывает четыре
корзины, interleaving и подсчёт остатка дневной нормы новизны; всё это —
WP-13. В WP-05 остаётся то, без чего сессии не существует вовсе:

- выборка `review_states` по сроку с предикатом обучаемости §14.3 **внутри**
  запроса, а не после выборки: §16 запрещает прямо — «иначе на материале с
  большим числом непригодных фактов планировщик перебирает их на каждом тике»;
- сортировка корзины 2 — `lapses DESC, due_at ASC`, «факты, на которых человек
  регулярно ошибается, идут первыми»;
- ограничение `users.session_size` (§3) и порог `MIN_SESSION_ITEMS` (§30.3).

Чего здесь нет: корзины 1 (просроченные более `OVERDUE_DAYS`), 3 (слабые темы)
и 4 (новые факты с остатком `new_per_day`), interleaving, тихие часы §17.1,
`daily_cap`, `MIN_PUSH_INTERVAL`, брошенные сессии §17.2 — **WP-13**. Новые
факты при этом не теряются: §4.6 ставит им `due_at = processed_at +
NEW_FACT_DELAY`, и они приходят той же выборкой по сроку.

Предикат обучаемости берётся обёрткой `teachable_facts_select`, а не вызовом
SQL-функции: иначе порог `MIN_FACT_CONFIDENCE` снова станет литералом, теперь
в коде планировщика. Запрет записан в `CLAUDE.md` и проверяется
`test_threshold_not_inlined_by_consumers`.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Final

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings
from core.db.models import Question, ReviewState, User
from core.db.teachable import teachable_facts_select
from core.learning.checking import ANSWERABLE_TYPES
from core.learning.selection import select_question
from core.logging import get_logger

log = get_logger(__name__)

DEFAULT_SESSION_SIZE: Final = 6
"""Размер сессии, когда у пользователя колонка пуста. §3, `users.session_size`.

Повторяет **умолчание схемы**, а не вводит новое число: `server_default` этой
колонки равен 6 (`0001_schema.py`). Пустой она бывает только у строк, созданных
в обход схемы, — и тогда правильнее взять то же значение, что взяла бы база,
чем придумать своё. В §30.3 этого числа нет и быть не должно: это поле
пользователя, а не порог конфигурации.

Расхождение с миграцией поймает `test_default_session_size_matches_the_schema`.
"""

PLANNER_VERSION: Final = "skeleton-1"
"""Значение `review_logs.planner_version` (§3, §13.1).

Как и `fsrs_version`, метка обязана быть отличимой: по ней видно, какие
строки журнала собраны минимальным планировщиком WP-05, а какие — настоящим
отбором §16 из WP-13.
"""


@dataclass(frozen=True)
class PlannedItem:
    """Факт и выбранный по §13.3 вопрос — заготовка элемента сессии."""

    fact_id: int
    question_id: int
    question_type: str | None
    direction: str | None


async def plan_items(
    session: AsyncSession,
    *,
    user_id: int,
    settings: Settings,
    now: datetime.datetime,
) -> list[PlannedItem]:
    """Задания для новой сессии. Пустой список — сессию создавать не нужно.

    Пустой список возвращается и когда кандидатов нет вовсе, и когда их меньше
    `MIN_SESSION_ITEMS`: §16 говорит, что в этом случае «сессия не создаётся и
    факты ждут следующего тика».
    """
    session_size = (
        await session.execute(
            select(func.coalesce(User.session_size, DEFAULT_SESSION_SIZE)).where(User.id == user_id)
        )
    ).scalar_one_or_none()
    if session_size is None:
        log.info("planner_without_user", user_id=user_id)
        return []

    teachable = teachable_facts_select(settings.MIN_FACT_CONFIDENCE).subquery()
    answerable = (
        exists()
        .where(
            Question.fact_id == ReviewState.fact_id,
            Question.status == "valid",
            Question.type.in_(sorted(ANSWERABLE_TYPES)),
        )
        .correlate(ReviewState)
    )

    candidates = (
        await session.execute(
            select(
                ReviewState.fact_id,
                ReviewState.last_q_type,
                ReviewState.last_direction,
            )
            .join(teachable, teachable.c.fact_id == ReviewState.fact_id)
            .where(
                ReviewState.user_id == user_id,
                # Дословно `NOT suspended`, как в §16. Форма условия решает,
                # применим ли частичный индекс `ix_review_states_user_due`:
                # его предикат — `WHERE NOT suspended` (§3.1), и доказать
                # импликацию Postgres умеет только для `NOT suspended` и
                # `suspended = false`. Замер на 200 тысячах строк с
                # `enable_seqscan = off`:
                #
                #   NOT suspended       -> Index Scan, due_at в Index Cond
                #   suspended = false   -> Index Scan, due_at в Index Cond
                #   suspended IS FALSE  -> Seq Scan, индекс не применён
                #
                # То есть `IS FALSE` теряет индекс так же, как `IS NOT TRUE`,
                # и §16 стоит выполнять буквально. Семантика при `NULL` та же:
                # `NOT NULL` даёт `NULL`, строка отбрасывается.
                ~ReviewState.suspended,
                ReviewState.due_at <= now,
                # Фильтр по отвечаемому типу стоит в запросе по двум причинам.
                # Первая — та же, что у предиката §16: иначе непригодные факты
                # перебираются на каждом тике. Вторая — `MIN_SESSION_ITEMS`
                # обязан считаться по заданиям, которые человек реально
                # получит, иначе сессия молча выходит короче порога.
                answerable,
            )
            .order_by(ReviewState.lapses.desc().nullslast(), ReviewState.due_at.asc())
            .limit(session_size)
        )
    ).all()

    items: list[PlannedItem] = []
    for fact_id, last_q_type, last_direction in candidates:
        chosen = await select_question(
            session,
            int(fact_id),
            last_q_type=last_q_type,
            last_direction=last_direction,
        )
        if chosen is None:
            # Защитная ветвь: `EXISTS` выше уже отобрал факты с валидным
            # отвечаемым вопросом. Разойтись они могут только при правке
            # одного из двух условий — и тогда факт молча выпал бы из сессии.
            log.info("candidate_without_question", user_id=user_id, fact_id=int(fact_id))
            continue
        items.append(
            PlannedItem(
                fact_id=int(fact_id),
                question_id=chosen.id,
                question_type=chosen.type,
                direction=chosen.direction,
            )
        )

    if len(items) < settings.MIN_SESSION_ITEMS:
        log.info(
            "session_not_assembled_too_few_candidates",
            user_id=user_id,
            candidates=len(items),
            minimum=settings.MIN_SESSION_ITEMS,
        )
        return []

    return items
