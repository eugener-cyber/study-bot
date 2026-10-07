"""Стадии обработки материала. ТЗ §4.1.

Перечень — единственный источник имён стадий. Имена попадают в
`materials.completed_stages`, то есть в базу, и строковые литералы, разложенные
по трём модулям, разойдутся: одно место будет писать `questions`, другое
искать `question`, и возобновление (§4.3) начнёт переисполнять готовую стадию.

Стадия описана записью, а не функцией, потому что §4.3 требует возобновления с
середины: чтобы начать с середины, нужно знать, что середина существует.
Цепочка вызовов этого не даёт.
"""

from __future__ import annotations

from dataclasses import dataclass

SKIPPED_SUFFIX = ":skipped"
"""Пропущенная стадия пишется как `questions:skipped`, а не `questions`.

§4.1: разница существенна. По ней возобновление отличает «стадия выполнена»
от «стадия сознательно пропущена», а кнопка «Достроить задания» знает, что
именно доделывать.
"""


@dataclass(frozen=True)
class Stage:
    """Одна стадия конвейера."""

    name: str
    skippable: bool = False
    """Пропускаема ли стадия. §4.1: единственная такая — `questions`, и только
    при `materials.mode = 'notes_only'` (§6.3)."""

    @property
    def skipped_marker(self) -> str:
        """Как пропуск этой стадии записывается в `completed_stages`."""
        return self.name + SKIPPED_SUFFIX


EXTRACT = Stage("extract")
SECTIONS = Stage("sections")
FACTS = Stage("facts")
NOTES = Stage("notes")
QUESTIONS = Stage("questions", skippable=True)
SCHEDULE = Stage("schedule")

ORDER: tuple[Stage, ...] = (EXTRACT, SECTIONS, FACTS, NOTES, QUESTIONS, SCHEDULE)
"""Порядок строгий (§4.1).

`schedule` обязательна и идёт последней: до неё у фактов нет состояния
повторения, и планировщик физически не может их увидеть (§4.6). Это и причина,
по которой её нельзя пометить пропускаемой даже в режиме `notes_only` —
там она выполняется и просто не создаёт ни одной строки.
"""

BY_NAME: dict[str, Stage] = {stage.name: stage for stage in ORDER}


def is_done(stage: Stage, completed: list[str]) -> bool:
    """Выполнена ли стадия — с учётом того, что пропуск тоже считается.

    §4.3: повторный запуск завершённой стадии — no-op. Пропущенная стадия
    завершённой считается: иначе возобновление материала `notes_only` каждый
    раз пыталось бы выполнить `questions` заново, а §6.3 требует, чтобы это
    делала только кнопка «Достроить задания».
    """
    return stage.name in completed or stage.skipped_marker in completed


def was_skipped(stage: Stage, completed: list[str]) -> bool:
    """Была ли стадия пропущена сознательно."""
    return stage.skipped_marker in completed


def next_stage(completed: list[str]) -> Stage | None:
    """Первая невыполненная стадия, или `None`, если выполнены все.

    §4.3: «воркер читает `completed_stages` и начинает с первой незавершённой
    стадии». Именно первой, а не следующей за последней выполненной: порядок в
    `completed_stages` не гарантирован, и опираться на него значило бы
    опираться на порядок вставки в массив.
    """
    for stage in ORDER:
        if not is_done(stage, completed):
            return stage
    return None


def reopen_for_questions(completed: list[str]) -> list[str]:
    """`completed_stages` для кнопки «Достроить задания» (§6.3).

    Снимает отметку о пропуске `questions` и отметку `schedule`, после чего
    обычное возобновление (§4.3) само начинает с `questions`. §6.3 так и
    написано: «запускает конвейер со стадии `questions` — то есть обычное
    возобновление по §4.3».

    Отдельного механизма возобновления «с заданной стадии» специально нет.
    Первая версия этого модуля его имела, и он не работал: пропущенная стадия
    считается выполненной, поэтому возобновление с `questions` возвращало
    пустой список, и кнопка не делала ничего. Два механизма возобновления
    разошлись бы и дальше — здесь остаётся один.

    `schedule` снимается тоже, и это обязательно: §4.6 создаёт строки
    `review_states` по обучаемым фактам, а до появления вопросов обучаемых
    фактов нет вовсе (§14.3). Повторный запуск безопасен — вставка
    идемпотентна по `ON CONFLICT (user_id, fact_id) DO NOTHING`.

    Ранние стадии (`extract`, `sections`, `facts`, `notes`) не трогаются:
    переисполнять их нечем и незачем, их артефакты уже есть.
    """
    dropped = {QUESTIONS.skipped_marker, QUESTIONS.name, SCHEDULE.name}
    return [name for name in completed if name not in dropped]
