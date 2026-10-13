"""Разбор ответа. ТЗ §22, §23.

Два места, которые легко сделать наполовину: подсказка локатора собирается по
**всем** фрагментам факта (§23: «для PDF важно именно каждого»), и разбор не
обращается к модели (§22: «Модель для объяснения не вызывается»). Первое
проверяется фактом на двух фрагментах, второе — отсутствием импорта слоя LLM
в самом модуле, а не фейком: фейк проверял бы только тот путь, который тест
прошёл.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.learning.explain import build_explanation, correct_answer_text, locator_hint
from tests.learning_factory import MCQ_PAYLOAD, add_fact, add_fragment, add_question, scaffold

pytestmark = pytest.mark.db

MODULE = Path(__file__).resolve().parent.parent / "core" / "learning" / "explain.py"


def test_explanation_never_reaches_the_llm_layer() -> None:
    """§22: разбор не вызывает модель.

    Проверяется по импортам модуля, а не фейком клиента: фейк закрывает лишь
    тот путь, по которому прошёл тест, а запрет относится ко всему модулю —
    `detail` пришёл вместе с фактом, и обращаться к модели в разборе незачем
    ни в одной ветке.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offending = {name for name in imported if name.startswith(("core.llm", "anthropic"))}
    assert not offending, f"разбор тянет слой модели: {sorted(offending)}"


@pytest.mark.parametrize(
    ("locator", "expected"),
    [
        ({"type": "web", "url": "file:///x.md", "quote": "Мел размечает"}, "«Мел размечает»"),
        ({"type": "web", "url": "file:///x.md"}, "file:///x.md"),
        ({"type": "web", "url": "file:///x.md", "quote": "   "}, "file:///x.md"),
        ({"type": "pdf", "page": 7}, "страница 7"),
        ({"type": "pdf"}, "PDF без номера страницы"),
        ({"type": "youtube", "id": "abc", "start": 12.0}, "локатор youtube"),
        (None, "источник не указан"),
        ({}, "источник не указан"),
    ],
)
def test_locator_hint_covers_every_variant_it_claims(
    locator: dict[str, object] | None, expected: str
) -> None:
    """§3.4: вариантов локатора шесть, реализованы два.

    Неизвестный вариант даёт **свой тип**, а не пустую строку: по нему видно,
    какого адаптера не хватает. Локатор `youtube` появится в WP-19 вместе с
    адаптером видео — до него разбор обязан остаться читаемым.
    """
    assert locator_hint(locator) == expected


def test_correct_answer_is_the_option_text_not_its_number() -> None:
    """§22: «Правильный ответ: …».

    Номер варианта через час после сессии ничего не говорит, а разбор читают
    именно потом — вместе с конспектом.
    """
    assert correct_answer_text("mcq", MCQ_PAYLOAD, {"correct_index": 1}, 1) == "утюг"


def test_correct_answer_falls_back_to_reference_when_the_index_is_broken() -> None:
    """Индекс вне диапазона — §11.1 и валидатор WP-10.

    До него такие строки в базе есть (Issue #33), и разбор обязан остаться
    читаемым: берётся `reference`, если он есть.
    """
    answer = {"correct_index": 9, "reference": "мел"}
    assert correct_answer_text("mcq", MCQ_PAYLOAD, answer, 9) == "мел"


def test_correct_answer_for_open_question_is_the_reference() -> None:
    """Открытый ответ §9.2 несёт эталон. Тип приедет в WP-15, поле есть уже."""
    assert correct_answer_text("open", {"question": "?"}, {"reference": "мел"}, None) == "мел"


def test_correct_answer_without_anything_is_a_dash() -> None:
    assert correct_answer_text("mcq", {"options": []}, {}, None) == "—"


async def test_explanation_quotes_every_fragment_of_the_fact(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """§23: подсказка собирается по **каждому** фрагменту факта.

    «Факт регулярно опирается на два-три фрагмента, и подсветка только
    первого показывает половину обоснования». Порядок — по `fragments.ord`,
    как и будет у подсветки страниц в WP-11.
    """
    base = await scaffold(engine)
    second = await add_fragment(engine, base, position=2, quote="Лён прочнее хлопка")
    fact_id = await add_fact(
        engine,
        base,
        statement="Мел и лён",
        detail="Оба утверждения опираются на разные абзацы источника.",
        fragment_ids=[base.fragment_id, second],
    )
    question_id = await add_question(engine, fact_id)

    async with sessions() as session:
        explanation = await build_explanation(
            session, question_id=question_id, verdict="correct", correct_option=0
        )

    assert len(explanation.locator_hints) == 2, "подсказка собрана не по всем фрагментам"
    assert explanation.locator_hints[0] == "«Мел размечает ткань по лицевой стороне.»"
    assert explanation.locator_hints[1] == "«Лён прочнее хлопка»"
    assert explanation.statement == "Мел и лён"
    assert explanation.detail.startswith("Оба утверждения")
    assert explanation.correct_answer == "мел"
    assert explanation.verdict == "correct"


async def test_explanation_survives_a_fact_without_fragments(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """Факт без evidence разбор не ломает.

    По И-1 такого факта быть не должно, и предикат §14.3 его не выдаёт — но
    разбор не то место, где об этом узнают: падение здесь означало бы, что
    человек не получил объяснение после ответа, который уже записан.
    """
    base = await scaffold(engine)
    fact_id = await add_fact(engine, base, fragment_ids=[])
    question_id = await add_question(engine, fact_id)

    async with sessions() as session:
        explanation = await build_explanation(
            session, question_id=question_id, verdict="wrong", correct_option=0
        )

    assert explanation.locator_hints == ()
    assert explanation.verdict == "wrong"
