"""Соответствие результата рейтингу. ТЗ §13.2 и оговорка §13.1.

Пороги §30.3 стоят здесь **литералами** 8000 и 45000 рядом с импортом из
`Settings`. Урок WP-01: тест, выраженный только через импортированную
константу, остаётся зелёным при любом её значении и нормы не проверяет.
"""

from __future__ import annotations

import pytest

from core.learning.rating import Rating, rating_for

FAST_MS = 8000
"""§30.3: `FAST_MS` = 8000. Литерал намеренно."""

SLOW_MS = 45000
"""§30.3: `SLOW_MS` = 45000. Литерал намеренно."""


def _rating(verdict: str, *, latency_ms: int = 20000, reps: int = 5) -> Rating:
    return rating_for(
        verdict, latency_ms=latency_ms, reps=reps, fast_ms=FAST_MS, slow_ms=SLOW_MS
    )


def test_spec_thresholds_match_settings() -> None:
    """Литералы теста совпадают с умолчаниями конфигурации.

    Так расхождение между §30.3 и `Settings` всплывает здесь, а не в тестах,
    которые про рейтинг ничего не утверждают.
    """
    from core.config import Settings

    assert Settings.model_fields["FAST_MS"].default == FAST_MS
    assert Settings.model_fields["SLOW_MS"].default == SLOW_MS


@pytest.mark.parametrize("verdict", ["wrong", "skipped"])
def test_wrong_and_dont_know_give_again(verdict: str) -> None:
    """§13.2: «неверно / „не знаю“ / пропуск» — Again."""
    assert _rating(verdict) is Rating.AGAIN


def test_partial_gives_hard() -> None:
    """§13.2: «частично верно» — Hard."""
    assert _rating("partial") is Rating.HARD


def test_correct_gives_good() -> None:
    """§13.2: «верно» — Good."""
    assert _rating("correct", latency_ms=20000) is Rating.GOOD


def test_correct_but_slow_gives_hard() -> None:
    """§13.2: «верно, но дольше SLOW_MS» — Hard."""
    assert _rating("correct", latency_ms=SLOW_MS + 1) is Rating.HARD


def test_correct_exactly_at_slow_threshold_is_not_slow_yet() -> None:
    """Граница: «дольше» читается строго, 45000 мс ещё не медленно."""
    assert _rating("correct", latency_ms=SLOW_MS) is Rating.GOOD


def test_correct_and_fast_gives_easy_when_reps_above_two() -> None:
    """§13.2: «верно, быстрее FAST_MS, и reps > 2» — Easy."""
    assert _rating("correct", latency_ms=FAST_MS - 1, reps=3) is Rating.EASY


def test_correct_and_fast_with_reps_two_is_only_good() -> None:
    """Граница `reps > 2`: при reps = 2 Easy не выдаётся.

    Проверка в обе стороны к предыдущему тесту: без неё условие `reps >= 2`
    прошло бы оба теста.
    """
    assert _rating("correct", latency_ms=FAST_MS - 1, reps=2) is Rating.GOOD


def test_correct_exactly_at_fast_threshold_is_not_fast() -> None:
    """Граница: «быстрее» читается строго, 8000 мс уже не быстро."""
    assert _rating("correct", latency_ms=FAST_MS, reps=5) is Rating.GOOD


def test_unknown_verdict_is_an_error_not_a_silent_again() -> None:
    """Неизвестный вердикт — ошибка, а не молчаливый Again.

    Молчаливый Again означал бы, что опечатка в вердикте портит расписание
    факта и выглядит при этом нормальным ответом.
    """
    with pytest.raises(ValueError, match="вердикт"):
        _rating("almost")
