"""Сборка данных для тестов педагогического слоя.

Один набор фабрик на все тесты WP-05: отбор, планировщик, сервис сессий,
сквозной путь. Четыре копии одного `INSERT` разошлись бы — п. 5 списка
костылей, — и разошлись бы именно в полях, которые проверяются: `direction` и
`shown_count` объявлены `nullable`, и половина тестов отбора существует ровно
ради пустых значений в них.

Фабрики принимают `engine` и пишут в своей транзакции: тесты отбора смотрят на
данные через собственные сессии, и видимость незакоммиченной вставки зависела
бы от порядка фикстур.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from core.db.models import Fact, Fragment, Material, Question, ReviewState, Section, User

NOW = datetime.datetime(2026, 10, 12, 12, 0, tzinfo=datetime.UTC)
"""Опорный момент тестов. Фиксированный, а не `now()`: расписание считается в
сутках, и тест, запущенный в полночь, иначе вёл бы себя иначе."""

MCQ_PAYLOAD: dict[str, Any] = {
    "question": "Какой инструмент размечает ткань?",
    "options": ["мел", "утюг", "ножницы"],
}
MCQ_ANSWER: dict[str, Any] = {"correct_index": 0}


@dataclass(frozen=True)
class Scaffold:
    """Пользователь, материал, секция и фрагмент — опора для факта."""

    user_id: int
    material_id: int
    section_id: int
    fragment_id: int


async def scaffold(engine: AsyncEngine, *, tg_id: int = 111) -> Scaffold:
    async with engine.begin() as connection:
        user_id = int(
            (
                await connection.execute(
                    insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
                )
            ).scalar_one()
        )
        material_id = int(
            (
                await connection.execute(
                    insert(Material)
                    .values(
                        user_id=user_id,
                        kind="text",
                        status="ready",
                        mode="full",
                        created_at=NOW,
                        processed_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )
        section_id = int(
            (
                await connection.execute(
                    insert(Section)
                    .values(material_id=material_id, ord=1, title="Раскрой")
                    .returning(Section.id)
                )
            ).scalar_one()
        )
        fragment_id = int(
            (
                await connection.execute(
                    insert(Fragment)
                    .values(
                        material_id=material_id,
                        ord=1,
                        kind="text",
                        locator={
                            "type": "web",
                            "url": "file:///materials/raskroy.md",
                            "quote": "Мел размечает ткань по лицевой стороне.",
                        },
                        text="Мел размечает ткань по лицевой стороне.",
                        quality=1.0,
                    )
                    .returning(Fragment.id)
                )
            ).scalar_one()
        )
    return Scaffold(
        user_id=user_id,
        material_id=material_id,
        section_id=section_id,
        fragment_id=fragment_id,
    )


async def add_fact(
    engine: AsyncEngine,
    base: Scaffold,
    *,
    statement: str = "Мел размечает ткань",
    detail: str = "Меловая линия видна на лицевой стороне и стирается щёткой.",
    confidence: float = 0.9,
    importance: int = 2,
    derived: bool = False,
    suspended: bool = False,
) -> int:
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(Fact)
                    .values(
                        material_id=base.material_id,
                        section_id=base.section_id,
                        statement=statement,
                        detail=detail,
                        kind="definition",
                        fragment_ids=[base.fragment_id],
                        importance=importance,
                        difficulty=2,
                        confidence=confidence,
                        derived=derived,
                        suspended=suspended,
                        generation_status="ok",
                        created_at=NOW,
                    )
                    .returning(Fact.id)
                )
            ).scalar_one()
        )


async def add_question(
    engine: AsyncEngine,
    fact_id: int,
    *,
    question_type: str = "mcq",
    direction: str | None = "forward",
    difficulty: int = 1,
    status: str = "valid",
    payload: dict[str, Any] | None = None,
    answer: dict[str, Any] | None = None,
    shown_count: int | None = 0,
    last_shown_at: datetime.datetime | None = None,
) -> int:
    """Вопрос. `direction` и `shown_count` можно задать пустыми намеренно.

    Пустые значения — не экзотика: колонки объявлены `nullable` (§3,
    `0001_schema.py:223` и `:234`), и на них ломается сортировка §13.3, если
    ключи не NULL-безопасны.
    """
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(Question)
                    .values(
                        fact_id=fact_id,
                        type=question_type,
                        direction=direction,
                        payload=payload if payload is not None else MCQ_PAYLOAD,
                        answer=answer if answer is not None else MCQ_ANSWER,
                        difficulty=difficulty,
                        status=status,
                        prompt_version="questions@v1",
                        model="manual",
                        provider="manual",
                        generation_attempt=1,
                        shown_count=shown_count,
                        last_shown_at=last_shown_at,
                        created_at=NOW,
                    )
                    .returning(Question.id)
                )
            ).scalar_one()
        )


async def add_review_state(
    engine: AsyncEngine,
    user_id: int,
    fact_id: int,
    *,
    due_at: datetime.datetime | None = None,
    reps: int = 0,
    lapses: int = 0,
    seen: bool = False,
    suspended: bool = False,
    last_q_type: str | None = None,
    last_direction: str | None = None,
    introduced_at: datetime.datetime | None = None,
    last_review: datetime.datetime | None = None,
) -> None:
    """Строка расписания. Умолчание — факт, которому пора (§16)."""
    async with engine.begin() as connection:
        await connection.execute(
            insert(ReviewState).values(
                user_id=user_id,
                fact_id=fact_id,
                due_at=due_at if due_at is not None else NOW - datetime.timedelta(minutes=1),
                state=0,
                reps=reps,
                lapses=lapses,
                seen=seen,
                suspended=suspended,
                last_q_type=last_q_type,
                last_direction=last_direction,
                introduced_at=introduced_at,
                last_review=last_review,
                created_at=NOW,
            )
        )
