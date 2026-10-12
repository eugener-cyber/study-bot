"""Планировщик повторений: интерфейс и заглушка WP-05. ТЗ §13.1.

Настоящий `py-fsrs` приезжает в **WP-12** и становится второй реализацией
этого же протокола: ни сервис ответа, ни хендлеры, ни их тесты от замены не
меняются. Ради этого интерфейс и существует с первого дня, хотя реализация
пока одна.

**Чего заглушка не делает.** Она не заполняет `stability`, `difficulty` и
`state`:

- `stability` и `difficulty` — §4.6 говорит прямо: «заполнит FSRS при первом
  ответе». Вывести стабильность из фиксированного интервала нельзя, а
  записанное число WP-16 принял бы за данные алгоритма при расчёте mastery
  (§14.1 считает `base` как `ln(1 + S)`).
- `state` — числовую шкалу задаёт `py-fsrs`, которого в зависимостях ещё нет.
  Ошибка в нумерации означала бы, что WP-12 прочитает «Review» как
  «Relearning» на фактах, где человек уже отвечал.

Следствие названо прямо: после ответов в базе будут строки с `reps > 0` при
`state = 0`, то есть «новый» факт с ответами. Это допустимо ровно потому, что
колонку `state` сегодня не читает никто, и перестанет быть допустимым в WP-12,
который заполнит её задним числом из `review_logs` — для этого журнал и
пишется целиком. Разбор решения — ADR-0006.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Protocol

from core.learning.rating import Rating

STUB_VERSION: Final = "stub-1"
"""Значение `review_logs.fsrs_version` для строк заглушки (§13.1).

Метка обязана быть отличимой: §13.1 требует, чтобы журнал делал возможным
пересчёт при смене алгоритма, и WP-12 начнёт именно с отбора строк `stub-1`.
"""

STUB_INTERVALS: Final[Mapping[Rating, timedelta]] = {
    Rating.AGAIN: timedelta(minutes=10),
    Rating.HARD: timedelta(days=1),
    Rating.GOOD: timedelta(days=3),
    Rating.EASY: timedelta(days=7),
}
"""Интервалы заглушки.

Модульные константы, а не поля `Settings`: §30.3 перечисляет настраиваемые
пороги, и этих среди них нет, а новое поле конфигурации упёрлось бы в
`test_no_config_field_missing_from_spec` и потребовало бы правки ТЗ под
временный код. Конфигурируемым должно быть то, что называет §30.3; интервалы
заглушки исчезают вместе с заглушкой в WP-12.

Значения выбраны так, чтобы сквозной путь было видно в один прогон: 10 минут
совпадает с `NEW_FACT_DELAY` (§4.6), остальные растут по рейтингу. Все
короче `MAX_INTERVAL_DAYS` (§13.1) — проверяется тестом.
"""


@dataclass(frozen=True)
class ReviewSnapshot:
    """Состояние факта **до** ответа. Собрано из колонок `review_states` §3.

    Из колонок, а не из того, что удобно заглушке: `py-fsrs` в WP-12 получает
    на вход ровно эти данные, и форма снимка не должна меняться вместе с
    реализацией.
    """

    reps: int
    lapses: int
    last_review: datetime | None
    stability: float | None
    difficulty: float | None
    state: int | None


@dataclass(frozen=True)
class ReviewOutcome:
    """Что записать после ответа.

    `None` в `stability`, `difficulty` и `state` означает «не трогать колонку»
    — сервис такие поля не пишет. Это и позволяет заглушке не придумывать
    чисел, а WP-12 — начать их отдавать, не меняя сервис.
    """

    due_at: datetime
    reps: int
    lapses: int
    last_review: datetime
    scheduled_days: int
    elapsed_days: int
    stability: float | None = None
    difficulty: float | None = None
    state: int | None = None


class Scheduler(Protocol):
    """Протокол планировщика повторений.

    `version` попадает в `review_logs.fsrs_version` на каждый ответ (§13.1).
    """

    @property
    def version(self) -> str: ...

    def review(
        self, snapshot: ReviewSnapshot, rating: Rating, *, now: datetime
    ) -> ReviewOutcome: ...


class FixedIntervalScheduler:
    """Заглушка: интервал зависит только от рейтинга. WP-05.

    Оговорка §13.1 о первом предъявлении реализована здесь, а не в WP-12,
    потому что иначе journal первых дней пришлось бы чинить задним числом:
    при `reps = 0` неверный ответ даёт Again, но `lapses` не растёт — «человек
    только впервые встретил факт, это не забывание».
    """

    @property
    def version(self) -> str:
        return STUB_VERSION

    def review(self, snapshot: ReviewSnapshot, rating: Rating, *, now: datetime) -> ReviewOutcome:
        interval = STUB_INTERVALS[rating]
        forgotten = rating is Rating.AGAIN and snapshot.reps > 0
        elapsed = (now - snapshot.last_review).days if snapshot.last_review else 0

        return ReviewOutcome(
            due_at=now + interval,
            reps=snapshot.reps + 1,
            lapses=snapshot.lapses + (1 if forgotten else 0),
            last_review=now,
            scheduled_days=interval.days,
            elapsed_days=max(elapsed, 0),
        )
