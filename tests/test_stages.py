"""Перечень стадий и возобновление. ТЗ §4.1, §4.3, §6.3.

Модуль без базы и без сети, поэтому маркер `db` здесь не нужен.
"""

from __future__ import annotations

import pytest

from core.ingest.stages import (
    BY_NAME,
    EXTRACT,
    ORDER,
    QUESTIONS,
    SCHEDULE,
    SKIPPED_SUFFIX,
    is_done,
    next_stage,
    reopen_for_questions,
    was_skipped,
)

NOTES_ONLY = ["extract", "sections", "facts", "notes", "questions:skipped", "schedule"]
FULL = ["extract", "sections", "facts", "notes", "questions", "schedule"]


def test_order_matches_spec() -> None:
    """§4.1 задаёт порядок буквально, и он единственный источник имён."""
    assert [stage.name for stage in ORDER] == [
        "extract",
        "sections",
        "facts",
        "notes",
        "questions",
        "schedule",
    ]


def test_only_questions_is_skippable() -> None:
    """§4.1: «Единственная стадия, которую допустимо пропустить, — `questions`».

    `schedule` не пропускаема даже в режиме `notes_only`: §6.3 требует, чтобы
    она выполнялась и просто не создавала ни одной строки `review_states`. До
    неё у фактов нет состояния повторения, и планировщик их не видит (§4.6).
    """
    assert [stage.name for stage in ORDER if stage.skippable] == ["questions"]


def test_skipped_marker_differs_from_name() -> None:
    """§4.1: «в `completed_stages` пишется `questions:skipped`, а не `questions`».

    Разница несёт смысл: по ней возобновление отличает «выполнено» от
    «пропущено сознательно».
    """
    assert QUESTIONS.skipped_marker == "questions" + SKIPPED_SUFFIX
    assert QUESTIONS.skipped_marker != QUESTIONS.name


def test_next_stage_on_empty_is_the_first() -> None:
    assert next_stage([]) is EXTRACT


def test_next_stage_skips_completed() -> None:
    assert next_stage(["extract", "sections"]) is not None
    assert next_stage(["extract", "sections"]).name == "facts"  # type: ignore[union-attr]


def test_next_stage_ignores_order_inside_completed() -> None:
    """Порядок внутри `completed_stages` не влияет на результат.

    §4.3 говорит «начинает с первой незавершённой стадии» — первой по порядку
    §4.1, а не следующей за последней записанной. Опираться на порядок в
    массиве значило бы опираться на порядок вставки.
    """
    assert next_stage(["sections", "extract"]) is not None
    assert next_stage(["sections", "extract"]).name == "facts"  # type: ignore[union-attr]


def test_fully_processed_material_has_no_next_stage() -> None:
    assert next_stage(FULL) is None


def test_notes_only_material_has_no_next_stage() -> None:
    """Пропущенная стадия считается завершённой.

    Иначе возобновление материала `notes_only` каждый раз пыталось бы
    выполнить `questions` заново, а §6.3 требует, чтобы это делала только
    кнопка «Достроить задания».
    """
    assert next_stage(NOTES_ONLY) is None


def test_was_skipped_distinguishes_skip_from_completion() -> None:
    assert was_skipped(QUESTIONS, NOTES_ONLY) is True
    assert was_skipped(QUESTIONS, FULL) is False


def test_is_done_counts_both_completion_and_skip() -> None:
    assert is_done(QUESTIONS, FULL) is True
    assert is_done(QUESTIONS, NOTES_ONLY) is True
    assert is_done(QUESTIONS, ["extract"]) is False


# --- Кнопка «Достроить задания» (§6.3) -------------------------------------


def test_reopen_resumes_at_questions() -> None:
    """Главное свойство кнопки: возобновление начинается с `questions`.

    Первая версия модуля имела отдельный механизм «возобновить с заданной
    стадии», и он не работал: пропущенная стадия считается выполненной,
    поэтому возобновление с `questions` возвращало пустой список и кнопка не
    делала ничего. Обнаружилось прогоном в консоли до написания тестов.
    """
    reopened = reopen_for_questions(NOTES_ONLY)
    assert next_stage(reopened) is QUESTIONS


def test_reopen_drops_schedule_too() -> None:
    """`schedule` снимается обязательно.

    §4.6 создаёт строки `review_states` по обучаемым фактам, а до появления
    вопросов обучаемых фактов нет вовсе (§14.3). Оставив `schedule`
    выполненной, кнопка достроила бы вопросы и не дала бы ни одного задания —
    то есть дефект был бы невидим: материал `ready`, вопросы есть, заданий
    нет.
    """
    reopened = reopen_for_questions(NOTES_ONLY)
    assert SCHEDULE.name not in reopened


def test_reopen_keeps_early_stages() -> None:
    """Ранние стадии не переисполняются.

    §6.3: кнопка «запускает конвейер со стадии `questions`». Потеря отметки
    `extract` означала бы повторное извлечение — для сканированного PDF это
    vision-вызов на каждую страницу, то есть повторная оплата самой дорогой
    стадии.
    """
    reopened = reopen_for_questions(NOTES_ONLY)
    for name in ("extract", "sections", "facts", "notes"):
        assert name in reopened


def test_reopen_is_idempotent() -> None:
    """Двойное нажатие кнопки даёт то же состояние.

    Кнопка остаётся на экране до перерисовки, и второе нажатие достижимо —
    тот же случай, что двойной тап по кнопке согласия в WP-01.
    """
    once = reopen_for_questions(NOTES_ONLY)
    twice = reopen_for_questions(once)
    assert once == twice


def test_reopen_on_full_material_also_rebuilds_questions() -> None:
    """На полном материале кнопка тоже начинает с `questions`.

    Фиксирую поведение: кнопка показывается только у `notes_only` (§6.3), но
    функция вызвана на полном материале даёт перегенерацию вопросов, а не
    ошибку. Это осознанно — иначе повторный вызов после уже достроенного
    материала падал бы, а гонка двух нажатий к этому и ведёт.
    """
    reopened = reopen_for_questions(FULL)
    assert next_stage(reopened) is QUESTIONS


def test_by_name_covers_every_stage() -> None:
    """Поиск по имени покрывает перечень целиком.

    Нужен возобновлению: `completed_stages` приходит из базы строками, и
    сопоставление с перечнем должно быть полным, иначе неизвестное имя
    молча проигнорируется.
    """
    assert set(BY_NAME) == {stage.name for stage in ORDER}
    assert len(BY_NAME) == len(ORDER)


@pytest.mark.parametrize("stage", ORDER, ids=[stage.name for stage in ORDER])
def test_every_stage_is_reachable_by_resuming(stage: object) -> None:
    """С любой стадии можно возобновиться, если предыдущие выполнены.

    Проверка полноты механизма §4.3: если бы какая-то стадия выпадала из
    перебора, материал застревал бы на ней без видимой причины.
    """
    index = ORDER.index(stage)  # type: ignore[arg-type]
    completed = [s.name for s in ORDER[:index]]
    assert next_stage(completed) is stage
