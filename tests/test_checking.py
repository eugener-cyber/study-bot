"""Программная проверка ответа §21.1 и реестр отвечаемых типов.

Главное здесь — не вердикт MCQ, а то, что множество `ANSWERABLE_TYPES`
**выведено** из реестра обработчиков. Отдельный список типов разошёлся бы с
реестром, и планировщик начал бы выдавать задание, на которое нечем ответить.
"""

from __future__ import annotations

import pytest

from core.learning.checking import (
    ANSWERABLE_TYPES,
    CHECKERS,
    InvalidAnswer,
    check_answer,
)

MCQ_PAYLOAD = {"question": "Что такое митоз?", "options": ["деление", "слияние", "рост"]}
MCQ_ANSWER = {"correct_index": 0}


def test_answerable_types_are_exactly_the_registry_keys() -> None:
    """Множество равно ключам реестра — не повторяет их, а выводится.

    Ассерт на равенство ключам, а не на литерал: литерал проверяет следующий
    тест, а этот — что второго списка типов в коде нет.
    """
    assert ANSWERABLE_TYPES == frozenset(CHECKERS)


def test_only_mcq_is_answerable_in_this_package() -> None:
    """В WP-05 отвечаемый тип один — `mcq`. Литерал намеренно.

    Issue #31: `open` добавляется в WP-15 вместе с двумя ступенями проверки,
    `match_text` и `match_image` — в WP-14 вместе с интерфейсом. Этот ассерт
    упадёт, если тип попал в реестр без обработчика ответа.
    """
    assert ANSWERABLE_TYPES == frozenset({"mcq"})
    assert "open" not in ANSWERABLE_TYPES


def test_correct_option_is_correct() -> None:
    result = check_answer("mcq", MCQ_PAYLOAD, MCQ_ANSWER, {"option": 0})
    assert result.verdict == "correct"
    assert result.correct_option == 0


def test_wrong_option_is_wrong() -> None:
    result = check_answer("mcq", MCQ_PAYLOAD, MCQ_ANSWER, {"option": 2})
    assert result.verdict == "wrong"
    assert result.correct_option == 0


def test_dont_know_is_skipped_not_wrong() -> None:
    """§13.2 различает «неверно» и «не знаю» только вердиктом.

    Рейтинг у них один — Again, — но `review_logs.verdict` хранится отдельно
    (§3), и по нему потом считается §25. Склеить их здесь значило бы потерять
    эту разницу навсегда.
    """
    result = check_answer("mcq", MCQ_PAYLOAD, MCQ_ANSWER, {"idk": True})
    assert result.verdict == "skipped"


def test_option_out_of_range_is_rejected() -> None:
    """Вариант вне диапазона — ошибка, а не «неверно».

    `callback_data` приходит от клиента (§19.1), и подставить туда любое число
    ничто не мешает. Засчитать это как неверный ответ значило бы испортить
    расписание факта по чужому запросу.
    """
    with pytest.raises(InvalidAnswer):
        check_answer("mcq", MCQ_PAYLOAD, MCQ_ANSWER, {"option": 7})


def test_answer_without_option_is_rejected() -> None:
    with pytest.raises(InvalidAnswer):
        check_answer("mcq", MCQ_PAYLOAD, MCQ_ANSWER, {})


def test_unanswerable_type_is_rejected_loudly() -> None:
    """Тип без обработчика ответа даёт ошибку, а не вердикт.

    Это вторая половина защиты: даже если такой вопрос как-то попал в сессию,
    молчаливого «неверно» не будет.
    """
    with pytest.raises(InvalidAnswer, match="open"):
        check_answer("open", {"question": "?"}, {"reference": "митоз"}, {"text": "митоз"})
