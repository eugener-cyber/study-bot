"""Разбор ответа. ТЗ §22, §23.

§22: «Модель для объяснения не вызывается: `detail` сгенерирован вместе с
фактом, поэтому разбор появляется мгновенно и всегда согласован с конспектом».
Здесь собираются **данные** разбора; рендер — у потребителя: бот рисует HTML
§19.2, консольный харнесс печатает текст. Так §конвенция «в `bot/` только
разбор апдейта, вызов сервиса и рендер» выполняется без второй копии разбора.

**Чего здесь нет.** Кнопка `📍 Показать в источнике` из разметки §22 не
рисуется: настоящих локаторов PDF не существует до WP-06, а рендер страницы с
подсветкой — WP-11. Вместо ссылки разбор несёт **подсказку локатора** —
цитату фрагмента или номер страницы, — то есть ровно то, что уже есть в
данных. Кнопка `[⚠️]` из той же разметки тоже не рисуется: жалобы — §24 и
WP-17.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import Fact, Fragment, Question
from core.learning.rating import Verdict


@dataclass(frozen=True)
class Explanation:
    """Разбор §22: вердикт, правильный ответ, «Почему» и подсказки локаторов."""

    verdict: Verdict
    statement: str
    """Утверждение факта — заголовок разбора и строка списка ошибок §22.1."""
    correct_answer: str
    detail: str
    """`facts.detail` — блок «Почему» §22."""
    locator_hints: tuple[str, ...]
    """По одной подсказке на каждый фрагмент факта.

    «Для PDF важно именно каждого: факт регулярно опирается на два-три
    фрагмента, и подсветка только первого показывает половину обоснования»
    (§23). До WP-11 это текст, но уже по всем фрагментам — иначе при переходе
    на подсветку обнаружилось бы, что цепочка собиралась по первому.
    """


def locator_hint(locator: Mapping[str, Any] | None) -> str:
    """Человекочитаемая подсказка из локатора §3.4.

    Варианты §3.4 перечислены не все: в WP-05 существуют только `web` (его
    отдаёт харнесс для текстовых файлов) и `pdf` (появится в WP-06). Для
    неизвестного варианта возвращается его тип, а не пустая строка: по типу
    видно, какого адаптера не хватает.
    """
    if not locator:
        return "источник не указан"

    kind = str(locator.get("type", "")) or "неизвестный"
    if kind == "web":
        quote = str(locator.get("quote") or "").strip()
        return f"«{quote}»" if quote else str(locator.get("url") or "веб-источник")
    if kind == "pdf":
        page = locator.get("page")
        return f"страница {page}" if page is not None else "PDF без номера страницы"
    return f"локатор {kind}"


def correct_answer_text(
    question_type: str | None,
    payload: Mapping[str, Any],
    answer: Mapping[str, Any],
    correct_option: int | None,
) -> str:
    """Строка «Правильный ответ» §22 для отвечаемых типов.

    Для `mcq` это текст варианта, а не его номер: номер в разборе через час
    после сессии ничего не говорит.
    """
    if question_type == "mcq" and correct_option is not None:
        options = list(payload.get("options") or [])
        if 0 <= correct_option < len(options):
            return str(options[correct_option])
    reference = answer.get("reference")
    return str(reference) if reference else "—"


async def build_explanation(
    session: AsyncSession,
    *,
    question_id: int,
    verdict: Verdict,
    correct_option: int | None,
) -> Explanation:
    """Собирает разбор по вопросу. Ни одного обращения к модели."""
    question_type, payload, answer, statement, detail, fragment_ids = (
        await session.execute(
            select(
                Question.type,
                Question.payload,
                Question.answer,
                Fact.statement,
                Fact.detail,
                Fact.fragment_ids,
            )
            .join(Fact, Fact.id == Question.fact_id)
            .where(Question.id == question_id)
        )
    ).one()

    locator_rows = (
        await session.execute(
            select(Fragment.locator)
            .where(Fragment.id.in_(list(fragment_ids or [])))
            .order_by(Fragment.ord)
        )
    ).all()
    locators: list[Mapping[str, Any] | None] = [row[0] for row in locator_rows]

    return Explanation(
        verdict=verdict,
        statement=str(statement),
        correct_answer=correct_answer_text(question_type, payload, answer, correct_option),
        detail=str(detail),
        locator_hints=tuple(locator_hint(item) for item in locators),
    )
