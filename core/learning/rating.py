"""Соответствие результата рейтингу FSRS. ТЗ §13.2.

Чистая функция без обращений к базе и к настройкам: пороги приходят
параметрами. Так её можно проверить таблицей §13.2 целиком, не поднимая
Postgres и не собирая `Settings`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final, Literal

Verdict = Literal["correct", "partial", "wrong", "skipped"]
"""Исход проверки ответа. `skipped` — явное «не знаю» (§20) и пропуск."""

VERDICTS: Final[frozenset[str]] = frozenset({"correct", "partial", "wrong", "skipped"})
"""Допустимые вердикты. Множество, а не проверка по месту: вердикт приходит из
трёх источников — программной проверки §21.1, судьи §21.3 и арбитража §21.4, —
и расхождение в написании одного из них иначе обнаружилось бы по испорченному
расписанию."""


class Rating(IntEnum):
    """Шкала FSRS. Числа заданы `py-fsrs` и в §13.2 названы словами.

    `IntEnum`, а не числа по коду: `rating` в `review_logs` хранится числом
    (§3), и литеральная двойка в коде ничего не говорит читателю.
    """

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


def rating_for(
    verdict: str,
    *,
    latency_ms: int | None,
    reps: int,
    fast_ms: int,
    slow_ms: int,
) -> Rating:
    """Рейтинг по таблице §13.2.

    `reps` — число ответов **до** текущего: условие Easy «`reps > 2`» говорит
    о накопленной истории факта, а не о текущем ответе.

    Границы читаются строго, как написано в §13.2: «дольше `SLOW_MS`» и
    «быстрее `FAST_MS`». Ответ ровно на пороге попадает в обычный случай.

    Соотнесения (§13.2, «менее половины пар — Again, от половины — Hard»)
    здесь не трактуются: вердикт `partial` уже несёт этот результат, а считает
    пары проверка ответа §21.1 — WP-14, где появляется сам тип задания.

    `latency_ms = None` означает «время неизвестно»: момент выдачи не записан.
    Тогда оба условия на скорость не применяются, и верный ответ даёт Good.
    Подставлять ноль было бы хуже — он означает «ответил мгновенно» и выдавал
    бы Easy за пропавшую отметку времени.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"неизвестный вердикт: {verdict!r}")

    if verdict in {"wrong", "skipped"}:
        return Rating.AGAIN
    if verdict == "partial":
        return Rating.HARD
    if latency_ms is None:
        return Rating.GOOD
    if latency_ms > slow_ms:
        return Rating.HARD
    if latency_ms < fast_ms and reps > 2:
        return Rating.EASY
    return Rating.GOOD
