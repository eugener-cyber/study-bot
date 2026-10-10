"""Схемы ответов модели. ТЗ §6.1, §7.1, §9.2, §10.1.

Примеры в тестах взяты **из текста спецификации**, а не построены из схем
кода: иначе тест сравнивал бы схему с собой и прошёл бы при схеме, которой §7.1
не соответствует вовсе.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from core.llm.estimation import StageCoefficients
from core.llm.schemas import (
    SCHEMA_BY_PURPOSE,
    FactsAnswer,
    McqQuestion,
    QuestionSet,
    SectionsAnswer,
)

# §7.1 дословно.
SPEC_FACTS: dict[str, Any] = {
    "title": "string",
    "summary_md": "3-8 предложений",
    "facts": [
        {
            "statement": "проверяемое утверждение одним предложением",
            "detail": "1-2 предложения раскрытия",
            "kind": "definition",
            "fragment_ids": [12, 13],
            "importance": 3,
            "difficulty": 2,
            "derived": False,
            "confidence": 0.9,
        }
    ],
}

# §9.2 дословно, payload и answer по типам.
SPEC_QUESTIONS: dict[str, Any] = {
    "questions": [
        {
            "type": "mcq",
            "direction": "forward",
            "difficulty": 2,
            "payload": {"question": "…", "options": ["A", "B", "C", "D"]},
            "answer": {"correct_index": 2},
        },
        {
            "type": "open",
            "direction": "reverse",
            "difficulty": 3,
            "payload": {"question": "…", "hint": None},
            "answer": {
                "reference": "эталон",
                "accept": ["синоним", "альт. формулировка"],
                "must_include": ["ключ1", "ключ2"],
            },
        },
    ]
}


def _fact(**overrides: Any) -> dict[str, Any]:
    return {**SPEC_FACTS, "facts": [{**SPEC_FACTS["facts"][0], **overrides}]}


def test_spec_example_of_7_1_parses() -> None:
    answer = FactsAnswer.model_validate(SPEC_FACTS)
    assert answer.facts[0].fragment_ids == [12, 13]
    assert answer.facts[0].kind == "definition"


def test_spec_example_of_9_2_parses_into_the_right_classes() -> None:
    answer = QuestionSet.model_validate(SPEC_QUESTIONS)
    assert isinstance(answer.questions[0], McqQuestion)
    assert answer.questions[0].answer.correct_index == 2


def test_extra_field_is_rejected() -> None:
    """§6.1 «строгая схема». Лишнее поле — ошибка, не мусор, который тихо отбросят.

    Тихо отбросить — худший вариант: §6.1 предусматривает повтор с
    приклеенным текстом ошибки, то есть механизм исправления уже есть, и он
    работает только если расхождение считается ошибкой.
    """
    with pytest.raises(ValidationError):
        FactsAnswer.model_validate({**SPEC_FACTS, "notes_md": "лишнее"})


def test_derived_has_no_default() -> None:
    """Пропуск `derived` — ошибка, а не «не выведен».

    По §7.3 проверке 4 выведенный факт не участвует в SRS. Умолчание `false`
    отправило бы догадку модели в обучение как подтверждённое знание — то
    есть нарушило бы §7.3 молча, без единой ошибки.
    """
    without = {key: value for key, value in SPEC_FACTS["facts"][0].items() if key != "derived"}
    with pytest.raises(ValidationError, match="derived"):
        FactsAnswer.model_validate({**SPEC_FACTS, "facts": [without]})


@pytest.mark.parametrize("field", ["importance", "difficulty", "confidence", "kind"])
def test_other_fact_fields_are_required_too(field: str) -> None:
    """Умолчаний нет ни у одного поля факта: пропуск стал бы утверждением."""
    without = {key: value for key, value in SPEC_FACTS["facts"][0].items() if key != field}
    with pytest.raises(ValidationError, match=field):
        FactsAnswer.model_validate({**SPEC_FACTS, "facts": [without]})


@pytest.mark.parametrize("value", [0, 4, -1])
def test_importance_outside_the_scale_is_rejected(value: int) -> None:
    """Шкала §3.2 — ровно 1..3. `SMALLINT` в §3 допускает больше, смысл имеют три."""
    with pytest.raises(ValidationError):
        FactsAnswer.model_validate(_fact(importance=value))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_outside_zero_one_is_rejected(value: float) -> None:
    """§7.1: 0..1. Выход за пределы ломает §7.3, где порог сравнивается с
    произведением `confidence × min(quality)`."""
    with pytest.raises(ValidationError):
        FactsAnswer.model_validate(_fact(confidence=value))


def test_confidence_boundaries_are_allowed() -> None:
    """Обе границы допустимы: 0 и 1 — законные значения, а не «вне диапазона».

    Без этой проверки прошёл бы и строгий интервал, и факт с уверенностью 1
    отбрасывался бы как ошибка схемы.
    """
    assert FactsAnswer.model_validate(_fact(confidence=0.0)).facts[0].confidence == 0.0
    assert FactsAnswer.model_validate(_fact(confidence=1.0)).facts[0].confidence == 1.0


def test_fact_without_evidence_is_rejected_by_the_schema() -> None:
    """Пустой `fragment_ids` — нарушение И-1 независимо от того, что подавали."""
    with pytest.raises(ValidationError):
        FactsAnswer.model_validate(_fact(fragment_ids=[]))


def test_unknown_fact_kind_is_rejected() -> None:
    """Вид факта — перечень §7.1. Неизвестный приехал бы в базу и выпал бы из §3.2."""
    with pytest.raises(ValidationError):
        FactsAnswer.model_validate(_fact(kind="мнение"))


def test_section_without_facts_is_allowed() -> None:
    """Титул, оглавление, список литературы — секции без проверяемых утверждений.

    Отвергать такой ответ значило бы требовать от модели выдумать факты там,
    где их нет.
    """
    assert FactsAnswer.model_validate({**SPEC_FACTS, "facts": []}).facts == []


def test_sections_answer_requires_at_least_one_section() -> None:
    with pytest.raises(ValidationError):
        SectionsAnswer.model_validate({"sections": []})


def test_section_without_fragments_is_rejected() -> None:
    """Секция без фрагментов не содержит ничего: по ней нельзя извлечь факт."""
    with pytest.raises(ValidationError):
        SectionsAnswer.model_validate({"sections": [{"title": "Пустая", "fragment_ids": []}]})


@pytest.mark.parametrize("count", [1, 5])
def test_question_set_size_follows_10_1(count: int) -> None:
    """§10.1 и §10.2 №1: 2–4 вопроса."""
    one = SPEC_QUESTIONS["questions"][0]
    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": [one] * count})


def test_mcq_options_limit_is_four() -> None:
    """§9.2: «вариантов mcq не более 4» — ограничение платформы.

    Текст кнопки до 64 символов, `callback_data` до 64 байт, поэтому длинные
    варианты нумеруются в теле сообщения.
    """
    five = {
        **SPEC_QUESTIONS["questions"][0],
        "payload": {"question": "…", "options": ["A", "B", "C", "D", "E"]},
    }
    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": [five, SPEC_QUESTIONS["questions"][1]]})


def test_match_text_requires_three_to_five_pairs() -> None:
    """§9.2: текстовое соотнесение 3–5 пар."""
    payload = {
        "type": "match_text",
        "direction": "forward",
        "difficulty": 2,
        "payload": {"prompt": "Соотнесите", "left": ["A", "B"], "right": ["1", "2"]},
        "answer": {"pairs": [[0, 1], [1, 0]]},
    }
    with pytest.raises(ValidationError):
        QuestionSet.model_validate({"questions": [payload, SPEC_QUESTIONS["questions"][1]]})


def test_correct_index_upper_bound_is_not_checked_here() -> None:
    """Границы массива — §11.1, то есть WP-10, и это сознательно.

    §11 отклоняет **один вопрос**, сохраняя остальные. Проверь здесь — и это
    стало бы ошибкой схемы, которая отвергает весь набор и расходует повтор
    §6.1 на то, для чего §11 предусматривает другую реакцию.
    """
    out_of_range = {
        **SPEC_QUESTIONS["questions"][0],
        "answer": {"correct_index": 99},
    }
    answer = QuestionSet.model_validate(
        {"questions": [out_of_range, SPEC_QUESTIONS["questions"][1]]}
    )
    assert isinstance(answer.questions[0], McqQuestion)
    assert answer.questions[0].answer.correct_index == 99


def test_discriminator_gives_one_error_not_four() -> None:
    """Дискриминатор по `type` (§9.1) нужен ради текста ошибки.

    Без него pydantic перечисляет ошибки всех четырёх вариантов, и
    приклеенный к повтору текст (§6.1) становится нечитаемым: модель получает
    четыре противоречащих замечания вместо одного.
    """
    broken = {**SPEC_QUESTIONS["questions"][0], "answer": {}}
    with pytest.raises(ValidationError) as error:
        QuestionSet.model_validate({"questions": [broken, SPEC_QUESTIONS["questions"][1]]})

    assert len(error.value.errors()) == 1
    assert error.value.errors()[0]["loc"][:3] == ("questions", 0, "mcq")


def test_unknown_question_type_is_rejected() -> None:
    """MVP §9.1 — четыре типа. `cloze` добавляется позже, «без изменения модели»."""
    with pytest.raises(ValidationError):
        QuestionSet.model_validate(
            {
                "questions": [
                    {**SPEC_QUESTIONS["questions"][0], "type": "cloze"},
                    SPEC_QUESTIONS["questions"][1],
                ]
            }
        )


def test_every_purpose_with_a_schema_has_an_output_ratio() -> None:
    """У каждого назначения есть и схема, и доля выхода для оценки §6.4.

    Иначе вызов проходил бы, а оценка его стоимости падала бы на неизвестном
    назначении — то есть учёт ломался бы на первом же вызове новой стадии.
    """
    coefficients = StageCoefficients()
    for purpose in SCHEMA_BY_PURPOSE:
        assert coefficients.output_ratio_for(purpose) > 0


def test_notes_has_no_schema_on_purpose() -> None:
    """Конспект §8 собирается из фактов, а не генерируется.

    Шаблон §8 ссылается на `section.summary_md` как на вход, остальные блоки
    выводит из сохранённых фактов, а отдельной таблицы под конспект §3 не
    содержит. Расхождение с §6.3, где `notes` учитывается как стадия с
    расходом токенов, вынесено change request'ом (#26).
    """
    assert "notes" not in SCHEMA_BY_PURPOSE
    assert set(SCHEMA_BY_PURPOSE) == {"sections", "facts", "questions"}
