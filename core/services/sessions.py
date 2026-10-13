"""Сеанс повторения: выдача задания, ответ, журнал, завершение.

ТЗ §13.1–13.3, §16, §18, §21.1, §22.1. Здесь живёт **весь** доступ к базе по
сессиям — §конвенция «доступ к БД через `core/services`». Транзакционная
граница одна и внешняя (`core/db/session.py`), коммита здесь нет.

**Почему мутация идёт условным `UPDATE`, а не проверкой перед записью.**
Postgres по умолчанию `READ COMMITTED`: две транзакции, прочитавшие
`answered_at IS NULL`, обе увидят пустое поле и обе запишут ответ. §18
требует, чтобы право на мутацию получил ровно один («первый callback получает
право на мутацию, второй видит заполненный `answered_at`»), поэтому проверка
встроена в `WHERE`, а право определяется `rowcount`. Там же, одним условием,
проверяется принадлежность элемента нажавшему: `callback_data` приходит от
клиента (§19.1), и подставить чужой `si` ничто не мешает — тот же довод, что
у `_owned_and_awaiting` в `bot/handlers/gate.py`.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings
from core.db.models import Fact, Question, ReviewLog, ReviewState, Session, SessionItem
from core.learning.checking import check_answer
from core.learning.explain import Explanation, build_explanation
from core.learning.planner import PLANNER_VERSION, PlannedItem, plan_items
from core.learning.rating import Rating, rating_for
from core.learning.scheduler import ReviewSnapshot, Scheduler
from core.logging import get_logger

log = get_logger(__name__)

SHOWN_AT = "shown_at"
"""Ключ отметки выдачи в `session_items.user_state`.

В `user_state`, а не в FSM: §19.1 требует, чтобы промежуточное состояние
задания переживало рестарт воркера, а у `redis` в §30.1 тома нет вовсе.
Отметка нужна для `latency_ms` (§3) и для рейтинга §13.2.
"""


@dataclass(frozen=True)
class ItemView:
    """Текущее задание — то, что показывают человеку."""

    item_id: int
    session_id: int
    question_id: int
    fact_id: int
    question_type: str
    payload: dict[str, Any]
    position: int
    total: int


@dataclass(frozen=True)
class SessionSummary:
    """Итог сеанса. §22.1: счёт, время, список ошибок."""

    total: int
    correct: int
    seconds: int
    mistakes: tuple[str, ...]


@dataclass(frozen=True)
class AnswerResult:
    """Что получилось из ответа: разбор и, если сеанс кончился, итог."""

    verdict: str
    rating: Rating
    explanation: Explanation
    finished: bool
    summary: SessionSummary | None


async def active_session_id(session: AsyncSession, user_id: int) -> int | None:
    """Идентификатор активной сессии пользователя или `None`.

    Активная ровно одна — это держит частичный уникальный индекс
    `uq_sessions_user_active` (§3.1, §38 №2), а не этот запрос.
    """
    return (
        await session.execute(
            select(Session.id).where(Session.user_id == user_id, Session.status == "active")
        )
    ).scalar_one_or_none()


async def assemble(
    session: AsyncSession,
    *,
    user_id: int,
    settings: Settings,
    now: datetime.datetime,
    kind: str = "manual",
) -> int | None:
    """Активная сессия пользователя: существующая или новая. `None` — заданий нет.

    Существующая возвращается раньше сборки, и это не оптимизация: §18
    допускает ровно одну активную сессию, и попытка собрать вторую упёрлась бы
    в уникальный индекс. `/review` при незавершённой сессии обязан показать
    текущее задание, а не завести второе.

    `None` означает «кандидатов меньше `MIN_SESSION_ITEMS`» или «нечего
    повторять» — по §16 это штатный исход, а не ошибка: «факты ждут следующего
    тика».
    """
    existing = await active_session_id(session, user_id)
    if existing is not None:
        return existing

    items = await plan_items(session, user_id=user_id, settings=settings, now=now)
    if not items:
        return None

    return await create_session(session, user_id=user_id, items=items, now=now, kind=kind)


async def create_session(
    session: AsyncSession,
    *,
    user_id: int,
    items: Sequence[PlannedItem],
    now: datetime.datetime,
    kind: str = "manual",
) -> int:
    """Создаёт сессию с элементами и возвращает её идентификатор.

    `kind = 'manual'` — сеанс, запрошенный человеком (`/review`) или
    харнессом; `'scheduled'` появится вместе с кроном §17 в WP-13.

    Вставка идёт под `SAVEPOINT`. При гонке (§18: «крон и `/review` могут
    стартовать одновременно») уникальный индекс поднимает `IntegrityError`, и
    без вложенного блока исключение отравило бы всю внешнюю транзакцию — то
    есть `/review` ответил бы ошибкой там, где §18 требует молча отказаться от
    создания и работать с существующей сессией.
    """
    if not items:
        raise ValueError("сессия без элементов не создаётся")

    try:
        async with session.begin_nested():
            session_id = int(
                (
                    await session.execute(
                        insert(Session)
                        .values(
                            user_id=user_id,
                            kind=kind,
                            status="active",
                            started_at=now,
                        )
                        .returning(Session.id)
                    )
                ).scalar_one()
            )
    except IntegrityError:
        existing = await active_session_id(session, user_id)
        if existing is None:  # pragma: no cover — индекс сработал, а сессии нет
            raise
        log.info("session_already_active", user_id=user_id, session_id=existing)
        return existing

    await session.execute(
        insert(SessionItem),
        [
            {
                "session_id": session_id,
                "ord": position,
                "question_id": item.question_id,
                "fact_id": item.fact_id,
            }
            for position, item in enumerate(items, start=1)
        ],
    )
    log.info("session_created", user_id=user_id, session_id=session_id, items=len(items))
    return session_id


async def session_of_item(session: AsyncSession, item_id: int, *, user_id: int) -> int | None:
    """Сессия задания, если она принадлежит этому человеку, иначе `None`.

    `user_id` стоит в условии, а не проверяется после выборки: идентификатор
    задания приходит из `callback_data`, то есть от клиента (§19.1), и чужая
    сессия не должна быть видна даже номером.
    """
    found = (
        await session.execute(
            select(SessionItem.session_id)
            .join(Session, Session.id == SessionItem.session_id)
            .where(SessionItem.id == item_id, Session.user_id == user_id)
        )
    ).scalar_one_or_none()
    return int(found) if found is not None else None


async def control_message_id(session: AsyncSession, session_id: int) -> int | None:
    """Сообщение с клавиатурой текущего задания, если оно известно. §20, §3."""
    return (
        await session.execute(select(Session.control_message_id).where(Session.id == session_id))
    ).scalar_one_or_none()


async def set_control_message(session: AsyncSession, session_id: int, message_id: int) -> None:
    """Запоминает сообщение с клавиатурой текущего задания. §20, §3.

    `sessions.control_message_id` — колонка §3 именно под это: по завершении
    задания клавиатуру нужно убрать или заменить разбором, а сделать это можно
    только у известного сообщения. Для медиагрупп §20 требует отдельного
    хранения идентификаторов в `session_items.user_state` — это WP-14 вместе с
    `match_image`.
    """
    await session.execute(
        update(Session).where(Session.id == session_id).values(control_message_id=message_id)
    )


async def current_item(session: AsyncSession, session_id: int) -> ItemView | None:
    """Первое неотвеченное задание сессии или `None`, если все отвечены."""
    total = (
        await session.execute(
            select(func.count())
            .select_from(SessionItem)
            .where(SessionItem.session_id == session_id)
        )
    ).scalar_one()

    row = (
        await session.execute(
            select(
                SessionItem.id,
                SessionItem.ord,
                SessionItem.question_id,
                SessionItem.fact_id,
                Question.type,
                Question.payload,
            )
            .join(Question, Question.id == SessionItem.question_id)
            .where(SessionItem.session_id == session_id, SessionItem.answered_at.is_(None))
            .order_by(SessionItem.ord)
            .limit(1)
        )
    ).first()

    if row is None:
        return None

    return ItemView(
        item_id=int(row[0]),
        session_id=session_id,
        question_id=int(row[2]),
        fact_id=int(row[3]),
        question_type=str(row[4]),
        payload=dict(row[5] or {}),
        position=int(row[1]),
        total=int(total),
    )


async def mark_shown(
    session: AsyncSession, item_id: int, *, user_id: int, now: datetime.datetime
) -> bool:
    """Отмечает задание выданным. Возвращает, случилось ли это сейчас.

    `user_id` обязателен и проверяется: отметка выдачи наращивает `shown_count`
    и переставляет `last_q_type`, а §13.3 сортирует вопросы по первому и
    смотрит на второй. Вызывать функцию на чужом задании сегодня неоткуда —
    идентификатор приходит из собственной выборки, а не от клиента, — но
    дешевле проверить здесь, чем полагаться на то, что так будет и дальше.

    §13.3: «После выдачи обновляются `last_q_type`, `last_direction`,
    `shown_count`, `last_shown_at`. Это счётчики презентации, а не learning
    state, поэтому И-4 они не нарушают». Обратное тоже обязано соблюдаться:
    `due_at`, `reps`, `lapses`, `seen` и `introduced_at` здесь не трогаются —
    И-4 запрещает менять learning state без ответа.

    Повторный показ того же задания (человек нажал `/review` дважды,
    сообщение потерялось) счётчики **не** наращивает: отметка выдачи стоит в
    `user_state`, и условие на её отсутствие стоит в `WHERE`. Иначе
    `shown_count` считал бы перезагрузки экрана, а §13.3 сортирует вопросы
    именно по нему.
    """
    row = (
        await session.execute(
            select(
                SessionItem.question_id,
                SessionItem.fact_id,
                SessionItem.user_state,
                Question.type,
                Question.direction,
                Session.user_id,
            )
            .join(Question, Question.id == SessionItem.question_id)
            .join(Session, Session.id == SessionItem.session_id)
            .where(SessionItem.id == item_id, Session.user_id == user_id)
        )
    ).first()
    if row is None:
        log.info("shown_not_marked_foreign_item", item_id=item_id, user_id=user_id)
        return False

    question_id, fact_id, user_state, question_type, direction, _ = row
    state = dict(user_state or {})
    state[SHOWN_AT] = now.isoformat()

    # Условие на отсутствие отметки стоит в `WHERE`, а не в `if` выше: между
    # чтением и записью может пройти второй показ, и тогда счётчик вырос бы
    # дважды за одну выдачу.
    # `CursorResult` вместо `Result`: `rowcount` объявлен только у первого, а
    # именно на нём держится право на мутацию (§18).
    marked: CursorResult[Any] = await session.execute(  # type: ignore[assignment]
        update(SessionItem)
        .where(
            SessionItem.id == item_id,
            SessionItem.user_state.is_(None) | ~SessionItem.user_state.has_key(SHOWN_AT),
        )
        .values(user_state=state)
    )
    if marked.rowcount != 1:
        return False

    await session.execute(
        update(Question)
        .where(Question.id == question_id)
        .values(
            shown_count=func.coalesce(Question.shown_count, 0) + 1,
            last_shown_at=now,
        )
    )
    await session.execute(
        update(ReviewState)
        .where(ReviewState.user_id == user_id, ReviewState.fact_id == fact_id)
        .values(last_q_type=question_type, last_direction=direction)
    )
    return True


async def answer_item(
    session: AsyncSession,
    *,
    user_id: int,
    item_id: int,
    raw: dict[str, Any],
    now: datetime.datetime,
    settings: Settings,
    scheduler: Scheduler,
) -> AnswerResult | None:
    """Принимает ответ. `None` — ответ не принят, и данные не изменились.

    `None` означает один из случаев §18: сессия уже не активна, на задание
    уже ответили (двойной тап), задания не существует, либо оно принадлежит
    другому человеку. Все четыре различимы только в журнале — наружу они
    выглядят одинаково, и это правильно: по разным ответам бота на эти случаи
    можно было бы перебирать чужие идентификаторы.

    Поэтому и порядок внутри: владелец проверяется **до** разбора ответа.
    Проверка ответа на чужом задании давала бы свой исход (`InvalidAnswerError`
    на испорченной строке вопроса, Issue #33), то есть отличала бы чужое
    существующее задание от несуществующего.
    """
    row = (
        await session.execute(
            select(
                SessionItem.session_id,
                SessionItem.question_id,
                SessionItem.fact_id,
                SessionItem.user_state,
                Question.type,
                Question.payload,
                Question.answer,
            )
            .join(Question, Question.id == SessionItem.question_id)
            .join(Session, Session.id == SessionItem.session_id)
            # Владелец проверяется **здесь**, до разбора ответа. Иначе чужое
            # существующее задание и несуществующее различимы снаружи: на
            # первом сначала срабатывала бы проверка ответа и давала свой
            # текст (испорченная строка вопроса, Issue #33), на втором —
            # отказ. Это оракул на существование чужих идентификаторов, и
            # закрыть его текстами в хендлере нельзя — только порядком.
            .where(SessionItem.id == item_id, Session.user_id == user_id)
        )
    ).first()
    if row is None:
        log.info("answer_rejected_unknown_or_foreign_item", item_id=item_id, user_id=user_id)
        return None

    session_id, question_id, fact_id, user_state, question_type, payload, answer = row
    check = check_answer(str(question_type), payload or {}, answer or {}, raw)

    latency_ms = _latency_ms(user_state, now, abandon_after_hours=settings.ABANDON_AFTER_HOURS)
    state_row = (
        await session.execute(
            select(
                ReviewState.reps,
                ReviewState.lapses,
                ReviewState.last_review,
                ReviewState.stability,
                ReviewState.difficulty,
                ReviewState.state,
            ).where(ReviewState.user_id == user_id, ReviewState.fact_id == fact_id)
        )
    ).first()
    if state_row is None:
        log.info("answer_rejected_no_review_state", item_id=item_id, fact_id=fact_id)
        return None

    snapshot = ReviewSnapshot(
        reps=int(state_row[0] or 0),
        lapses=int(state_row[1] or 0),
        last_review=state_row[2],
        stability=state_row[3],
        difficulty=state_row[4],
        state=state_row[5],
    )
    rating = rating_for(
        check.verdict,
        latency_ms=latency_ms,
        reps=snapshot.reps,
        fast_ms=settings.FAST_MS,
        slow_ms=settings.SLOW_MS,
    )

    # Право на мутацию: ровно один исполнитель получает rowcount = 1.
    claimed: CursorResult[Any] = await session.execute(  # type: ignore[assignment]
        update(SessionItem)
        .where(
            SessionItem.id == item_id,
            SessionItem.answered_at.is_(None),
            SessionItem.session_id.in_(
                select(Session.id).where(
                    Session.id == session_id,
                    Session.user_id == user_id,
                    Session.status == "active",
                )
            ),
        )
        .values(
            answer_raw=raw,
            verdict=check.verdict,
            rating=int(rating),
            answered_at=now,
            latency_ms=latency_ms,
        )
    )
    if claimed.rowcount != 1:
        log.info("answer_rejected_stale", item_id=item_id, user_id=user_id)
        return None

    outcome = scheduler.review(snapshot, rating, now=now)
    values: dict[str, Any] = {
        "due_at": outcome.due_at,
        "reps": outcome.reps,
        "lapses": outcome.lapses,
        "last_review": outcome.last_review,
        "seen": True,
        # §16.1: проставляется ровно один раз, в тот же момент, когда `seen`
        # переводится в true. COALESCE, а не проверка в коде: второй ответ по
        # тому же факту не должен сдвигать дату введения.
        "introduced_at": func.coalesce(ReviewState.introduced_at, now),
    }
    # Поля, которых заглушка не знает, не пишутся вовсе: см. докстроку
    # `core/learning/scheduler.py`. WP-12 начнёт их отдавать, и сервис
    # запишет их, не меняясь.
    for column, value in (
        ("stability", outcome.stability),
        ("difficulty", outcome.difficulty),
        ("state", outcome.state),
    ):
        if value is not None:
            values[column] = value

    await session.execute(
        update(ReviewState)
        .where(ReviewState.user_id == user_id, ReviewState.fact_id == fact_id)
        .values(**values)
    )

    await session.execute(
        insert(ReviewLog).values(
            user_id=user_id,
            fact_id=fact_id,
            question_id=question_id,
            rating=int(rating),
            verdict=check.verdict,
            scheduled_days=outcome.scheduled_days,
            elapsed_days=outcome.elapsed_days,
            latency_ms=latency_ms,
            fsrs_version=scheduler.version,
            planner_version=PLANNER_VERSION,
            reviewed_at=now,
        )
    )

    explanation = await build_explanation(
        session,
        question_id=question_id,
        verdict=check.verdict,
        correct_option=check.correct_option,
    )

    finished = await _finish_if_complete(session, session_id, now=now)
    summary = await session_summary(session, session_id) if finished else None

    log.info(
        "answer_accepted",
        session_id=session_id,
        item_id=item_id,
        verdict=check.verdict,
        rating=int(rating),
        finished=finished,
    )
    return AnswerResult(
        verdict=check.verdict,
        rating=rating,
        explanation=explanation,
        finished=finished,
        summary=summary,
    )


async def session_summary(session: AsyncSession, session_id: int) -> SessionSummary:
    """Итог сеанса §22.1: счёт, время и список ошибок.

    Обновлённая mastery затронутых тем и кнопки «Повторить ошибки» и
    «К конспекту» из §22.1 здесь не собираются: mastery — WP-16, быстрое
    повторение — §27 и WP-17, конспект — §8.1 и WP-08.
    """
    total, correct = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(SessionItem.verdict == "correct"),
            )
            .select_from(SessionItem)
            .where(SessionItem.session_id == session_id)
        )
    ).one()

    started_at, finished_at = (
        await session.execute(
            select(Session.started_at, Session.finished_at).where(Session.id == session_id)
        )
    ).one()

    mistakes = (
        (
            await session.execute(
                select(Fact.statement)
                .join(SessionItem, SessionItem.fact_id == Fact.id)
                .where(
                    SessionItem.session_id == session_id,
                    SessionItem.verdict.in_(["wrong", "skipped", "partial"]),
                )
                .order_by(SessionItem.ord)
            )
        )
        .scalars()
        .all()
    )

    seconds = 0
    if started_at is not None and finished_at is not None:
        seconds = max(int((finished_at - started_at).total_seconds()), 0)

    return SessionSummary(
        total=int(total),
        correct=int(correct),
        seconds=seconds,
        mistakes=tuple(str(item) for item in mistakes),
    )


async def _finish_if_complete(
    session: AsyncSession, session_id: int, *, now: datetime.datetime
) -> bool:
    """Переводит сессию в `finished`, когда неотвеченных заданий не осталось."""
    remaining = (
        await session.execute(
            select(func.count())
            .select_from(SessionItem)
            .where(SessionItem.session_id == session_id, SessionItem.answered_at.is_(None))
        )
    ).scalar_one()
    if remaining:
        return False

    await session.execute(
        update(Session)
        .where(Session.id == session_id, Session.status == "active")
        .values(status="finished", finished_at=now)
    )
    return True


def _latency_ms(
    user_state: dict[str, Any] | None,
    now: datetime.datetime,
    *,
    abandon_after_hours: int,
) -> int | None:
    """Время от выдачи до ответа. `None`, если его нельзя считать осмысленным.

    `None`, а не ноль: ноль означает «ответил мгновенно» и по §13.2 давал бы
    Easy за пропавшую отметку времени — то есть удлинял интервал на ошибке
    учёта.

    **Замер старше `ABANDON_AFTER_HOURS` тоже `None`.** §17.2 говорит, что
    сеанс без ответа дольше этого срока становится `abandoned` и FSRS не
    меняет; механизм живёт в WP-13, и до него такой сеанс остаётся
    отвечаемым. Живой прогон WP-05 это и показал: задание, выданное сутками
    раньше, дало `latency` в 87 157 486 мс, §13.2 прочитал его как «верно, но
    дольше `SLOW_MS`» и поставил Hard — то есть расписание факта испортило
    время простоя, а не ответ человека.

    Решение Архитектора от 2026-10-13: отвечать на такое задание можно,
    рейтинг ставится только по правильности, `latency_ms` в журнале остаётся
    пустым. Порог взят тот же, что у §17.2, — другого числа для «сеанс
    заброшен» в §30.3 нет, и выдумывать второе значило бы развести их при
    первой же правке.
    """
    raw = (user_state or {}).get(SHOWN_AT)
    if not isinstance(raw, str):
        return None
    try:
        shown_at = datetime.datetime.fromisoformat(raw)
    except ValueError:
        log.info("shown_at_unparsable", value=raw)
        return None
    if shown_at.tzinfo is None:
        # Отметка без часового пояса: вычитание из `now` с поясом даёт
        # `TypeError`, то есть ответ не был бы принят из-за косметики. Ревью
        # PR #34 показало, что прежний `except TypeError` стоял не там, где
        # исключение возникает: не-строку отсекает `isinstance` выше, а
        # падает **вычитание** после `try`. Проверка явная, а не перехват:
        # у наивной отметки нет верной трактовки — часовой пояс неизвестен, и
        # домыслить его значило бы вписать в журнал выдуманную длительность.
        log.info("shown_at_without_timezone", value=raw)
        return None
    elapsed = now - shown_at
    if elapsed > datetime.timedelta(hours=abandon_after_hours):
        log.info(
            "shown_at_too_old",
            hours=round(elapsed.total_seconds() / 3600, 1),
            limit_hours=abandon_after_hours,
        )
        return None
    return max(int(elapsed.total_seconds() * 1000), 0)
