"""Выбор вопроса в момент выдачи. ТЗ §13.3, инвариант И-2.

«Планируется знание, а не вопрос»: расписание висит на факте, а какой именно
вопрос задать, решается здесь и сейчас.

§13.3 описывает четыре шага отката; они собраны в **один** запрос — тремя
булевыми ключами сортировки, а не четырьмя обращениями к базе. Четыре запроса
дали бы четыре round-trip на каждый элемент сессии и разъехались бы в условиях
при первой правке.

**Все ключи NULL-безопасны по колонкам, а не только по привязкам.** `direction`
и `shown_count` объявлены `nullable` (§3), а в Postgres `ORDER BY … DESC` по
умолчанию `NULLS FIRST`: выражение `direction <> :last_dir` на строке с пустым
направлением даёт `NULL`, и такая строка обходит ту, которая шагу 1
удовлетворяет — молча. Поэтому `IS DISTINCT FROM`, который `NULL` не
возвращает никогда, и `COALESCE` на счётчике показов, где `ASC` дал бы
`NULLS LAST`, то есть прочитал бы «не показывали» как «показывали больше
всех». Обе ловушки держатся тестами, которые красны на наивной формулировке.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Final

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import Question
from core.learning.checking import ANSWERABLE_TYPES

COLD_START_DIFFICULTY_CAP: Final = 1
"""Потолок сложности задания, пока mastery не считается. §15, первая строка.

Не «3, то есть без ограничения»: §14.1 задаёт вырожденный случай прямо — при
`seen_facts = 0` mastery равна нулю, — а §15 при `mastery < 0.4` предписывает
потолок 1. Вычисленная mastery приходит в WP-16 и встаёт на место этой
константы, не меняя подписи `select_question`.

Факт при этом не остаётся без заданий: шаг 4 §13.3 снимает условие сложности,
когда под ним пусто.
"""


@dataclass(frozen=True)
class SelectedQuestion:
    """Выбранный вопрос. `type` и `direction` нужны для §13.3 на следующем шаге."""

    id: int
    type: str | None
    direction: str | None


async def select_question(
    session: AsyncSession,
    fact_id: int,
    *,
    last_q_type: str | None,
    last_direction: str | None,
    difficulty_cap: int = COLD_START_DIFFICULTY_CAP,
    answerable_types: Collection[str] = ANSWERABLE_TYPES,
) -> SelectedQuestion | None:
    """Вопрос по факту или `None`, если валидных отвечаемых вопросов нет.

    `None` — нормальный исход, а не ошибка: у факта может не быть ни одного
    вопроса отвечаемого типа (Issue #31), и планировщик такой факт пропускает.

    `answerable_types` — параметр, а не обращение к константе внутри: WP-14 и
    WP-15 расширяют множество вместе со своими обработчиками ответа, а тесты
    §13.3 проверяют шаги 1 и 2, для которых нужны два разных типа.
    """
    within_cap = Question.difficulty <= difficulty_cap
    other_type = Question.type.is_distinct_from(last_q_type)
    other_direction = Question.direction.is_distinct_from(last_direction)

    statement = (
        select(Question.id, Question.type, Question.direction)
        .where(
            Question.fact_id == fact_id,
            Question.status == "valid",
            Question.type.in_(sorted(answerable_types)),
        )
        .order_by(
            # Шаг 1 §13.3: другой тип, другое направление, в пределах потолка.
            and_(other_type, other_direction, within_cap).desc(),
            # Шаг 2: то же без условия на направление.
            and_(other_type, within_cap).desc(),
            # Шаг 3: то же без условия на тип. Потолок снимается последним —
            # §13.3 после правки v3.5: при конфликте приоритет у потолка.
            within_cap.desc(),
            # Шаг 4 — это уже отсутствие условий, остаётся порядок §13.3.
            func.coalesce(Question.shown_count, 0).asc(),
            Question.last_shown_at.asc().nullsfirst(),
        )
        .limit(1)
    )

    row = (await session.execute(statement)).first()
    if row is None:
        return None
    return SelectedQuestion(id=int(row[0]), type=row[1], direction=row[2])
