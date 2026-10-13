"""Заглушка планировщика повторений. WP-05, замена на `py-fsrs` — WP-12.

Проверяется ровно то, за что заглушка отвечает: монотонность интервалов,
оговорка §13.1 о первом предъявлении, потолок §13.1 и — отдельно — что она
**не придумывает** полей, смысл которых установит WP-12.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.learning.rating import Rating
from core.learning.scheduler import STUB_VERSION, FixedIntervalScheduler, ReviewSnapshot

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
MAX_INTERVAL_DAYS = 180
"""§30.3: `MAX_INTERVAL_DAYS` = 180. Литерал намеренно."""


def _snapshot(
    *, reps: int = 3, lapses: int = 0, last_review: datetime | None = None
) -> ReviewSnapshot:
    return ReviewSnapshot(
        reps=reps,
        lapses=lapses,
        last_review=last_review,
        stability=None,
        difficulty=None,
        state=0,
    )


@pytest.fixture
def scheduler() -> FixedIntervalScheduler:
    return FixedIntervalScheduler()


def test_version_is_honest_about_being_a_stub(scheduler: FixedIntervalScheduler) -> None:
    """Версия пишется в каждую строку `review_logs` (§13.1).

    Метка обязана отличать строки заглушки от строк FSRS: WP-12 пересчитывает
    расписание по журналу, и неотличимые строки он принял бы за данные
    алгоритма.
    """
    assert scheduler.version == STUB_VERSION
    assert "stub" in scheduler.version


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (Rating.AGAIN, timedelta(minutes=10)),
        (Rating.HARD, timedelta(days=1)),
        (Rating.GOOD, timedelta(days=3)),
        (Rating.EASY, timedelta(days=7)),
    ],
)
def test_interval_grows_with_rating(
    scheduler: FixedIntervalScheduler, rating: Rating, expected: timedelta
) -> None:
    outcome = scheduler.review(_snapshot(), rating, now=NOW)
    assert outcome.due_at == NOW + expected
    assert outcome.due_at > NOW


def test_longest_interval_stays_within_the_ceiling(scheduler: FixedIntervalScheduler) -> None:
    """§13.1: предельный интервал `MAX_INTERVAL_DAYS`.

    Заглушка его не вычисляет, но и не нарушает: проверка стоит здесь, чтобы
    правка интервалов «на подольше» не прошла молча.
    """
    longest = max(scheduler.review(_snapshot(), rating, now=NOW).due_at for rating in Rating)
    assert longest - NOW <= timedelta(days=MAX_INTERVAL_DAYS)


def test_first_failure_does_not_count_as_a_lapse(scheduler: FixedIntervalScheduler) -> None:
    """§13.1: при `reps = 0` неверный ответ даёт Again, но `lapses` не растёт.

    «Человек только впервые встретил факт, это не забывание».
    """
    outcome = scheduler.review(_snapshot(reps=0, lapses=0), Rating.AGAIN, now=NOW)
    assert outcome.reps == 1
    assert outcome.lapses == 0


def test_later_failure_does_count_as_a_lapse(scheduler: FixedIntervalScheduler) -> None:
    """Проверка в обе стороны: при `reps = 1` тот же Again `lapses` растит.

    Без этого теста условие «никогда не растить `lapses`» прошло бы
    предыдущий.
    """
    outcome = scheduler.review(_snapshot(reps=1, lapses=0), Rating.AGAIN, now=NOW)
    assert outcome.reps == 2
    assert outcome.lapses == 1


def test_correct_answer_never_adds_a_lapse(scheduler: FixedIntervalScheduler) -> None:
    outcome = scheduler.review(_snapshot(reps=1, lapses=2), Rating.GOOD, now=NOW)
    assert outcome.lapses == 2


def test_stub_invents_neither_stability_nor_difficulty_nor_state(
    scheduler: FixedIntervalScheduler,
) -> None:
    """Заглушка не заполняет поля, смысл которых задаёт WP-12.

    `stability` и `difficulty` — §4.6 прямо говорит «заполнит FSRS при первом
    ответе»; вывести стабильность из фиксированного интервала нельзя, а
    записанное число WP-16 принял бы за данные алгоритма при расчёте mastery.
    `state` — шкала определяется вместе с зависимостью `py-fsrs`, и ошибка в
    нумерации означала бы, что WP-12 прочитает «Review» как «Relearning».
    """
    outcome = scheduler.review(_snapshot(), Rating.GOOD, now=NOW)
    assert outcome.stability is None
    assert outcome.difficulty is None
    assert outcome.state is None


def test_elapsed_days_counts_from_last_review(scheduler: FixedIntervalScheduler) -> None:
    """`elapsed_days` в журнале §3 — промежуток с прошлого ответа."""
    outcome = scheduler.review(_snapshot(last_review=NOW - timedelta(days=4)), Rating.GOOD, now=NOW)
    assert outcome.elapsed_days == 4


def test_elapsed_days_is_zero_on_the_first_answer(scheduler: FixedIntervalScheduler) -> None:
    outcome = scheduler.review(_snapshot(reps=0), Rating.GOOD, now=NOW)
    assert outcome.elapsed_days == 0


def test_scheduled_days_matches_the_chosen_interval(scheduler: FixedIntervalScheduler) -> None:
    """`scheduled_days` — то, на сколько суток отложено, а не округление вверх.

    Интервал Again короче суток, и ноль здесь честнее единицы: WP-12
    пересчитывает по этому полю.
    """
    assert scheduler.review(_snapshot(), Rating.GOOD, now=NOW).scheduled_days == 3
    assert scheduler.review(_snapshot(), Rating.AGAIN, now=NOW).scheduled_days == 0
