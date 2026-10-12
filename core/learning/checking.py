"""Программная проверка ответа и реестр отвечаемых типов. ТЗ §21.1.

§21.1: «MCQ и соотнесения — программно, без вызова модели». В WP-05 из четырёх
типов §9.1 обработан один — `mcq`; соотнесения приезжают с интерфейсом в
WP-14, открытый ответ — с двумя ступенями и судьёй в WP-15 (§21.2–21.4).

**`ANSWERABLE_TYPES` выводится из `CHECKERS`, а не объявляется рядом.** Это не
стилистика: планировщик отбирает вопросы по этому множеству (§13.3), и
разойдись оно с реестром — человек получил бы задание, на которое нечем
ответить, а сессия встала бы на первом же элементе. Выведенное множество
изменить, не добавив обработчик, невозможно. Ограничение и сроки его снятия —
Issue #31.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from core.learning.rating import Verdict


class InvalidAnswerError(Exception):
    """Ответ не разобрать: неизвестный тип, нет варианта, вариант вне диапазона.

    Отдельное исключение, а не вердикт `wrong`: §13.2 превращает `wrong` в
    рейтинг Again и портит расписание факта. Ответ, который не разобран, не
    является неверным ответом человека — чаще всего это подделанный
    `callback_data` (§19.1) или рассогласование версий.
    """


@dataclass(frozen=True)
class CheckResult:
    """Исход проверки: вердикт §21.1 и то, что показать в разборе §22."""

    verdict: Verdict
    correct_option: int | None = None
    """Индекс правильного варианта MCQ — для строки «Правильный ответ» §22."""


Checker = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], CheckResult]
"""Проверка одного типа: `(payload, answer, raw) -> CheckResult`.

`payload` и `answer` — колонки `questions` (§9.2), `raw` — то, что прислал
человек, в форме, которую пишут в `session_items.answer_raw`.
"""


def check_mcq(
    payload: Mapping[str, Any],
    answer: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> CheckResult:
    """MCQ: сравнение с `correct_index`. §21.1, §9.2.

    «Не знаю» (§20, `v = -1` в фабрике `Ans`) приходит сюда как `idk` и даёт
    `skipped`, а не `wrong`: рейтинг у них одинаковый (§13.2), но в журнале
    они различаются, и на этой разнице стоят метрики §25.
    """
    stored = answer.get("correct_index")
    if not isinstance(stored, int) or isinstance(stored, bool):
        # Испорченная строка вопроса: §9.2 требует `correct_index`, а §11.1
        # относит проверку его границ к валидатору WP-10. До него такие строки
        # в базе есть — например, заглушки стадий из WP-03, оставшиеся до
        # подключения настоящих вызовов. Падать `KeyError` здесь нельзя:
        # наружу это выходит как «что-то сломалось», и причина теряется.
        raise InvalidAnswerError(
            f"в вопросе нет correct_index: {dict(answer)!r} — строка испорчена (§9.2)"
        )
    correct = stored

    if raw.get("idk"):
        return CheckResult(verdict="skipped", correct_option=correct)

    option = raw.get("option")
    if not isinstance(option, int):
        raise InvalidAnswerError(f"в ответе нет варианта: {dict(raw)!r}")

    options = payload.get("options") or []
    if not 0 <= option < len(options):
        raise InvalidAnswerError(f"вариант вне диапазона: {option} при {len(options)} вариантах")

    verdict: Verdict = "correct" if option == correct else "wrong"
    return CheckResult(verdict=verdict, correct_option=correct)


CHECKERS: Final[dict[str, Checker]] = {"mcq": check_mcq}
"""Реестр: тип задания → его проверка ответа.

Пополняется пакетом, который приносит и интерфейс типа, и проверку: WP-14 —
`match_text` и `match_image`, WP-15 — `open`. Пополнить его, не написав
обработчик, нельзя по устройству словаря.
"""

ANSWERABLE_TYPES: Final[frozenset[str]] = frozenset(CHECKERS)
"""Типы, на которые человек может ответить **сегодня**. §13.3 отбирает по нему.

Выведено из `CHECKERS` намеренно — см. докстроку модуля и Issue #31.
"""


def check_answer(
    question_type: str,
    payload: Mapping[str, Any],
    answer: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> CheckResult:
    """Проверяет ответ обработчиком своего типа.

    Неизвестный тип — `InvalidAnswerError`, а не «неверно»: см. докстроку
    исключения.
    """
    checker = CHECKERS.get(question_type)
    if checker is None:
        raise InvalidAnswerError(f"на задание типа {question_type!r} ответ ещё не проверяется")
    return checker(payload, answer, raw)
