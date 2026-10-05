"""Модель данных. ТЗ §3 — таблицы, §3.1 — индексы.

Объявления следуют §3 **буквально**, включая то, что там не сказано. Там, где
§3 молчит, умолчание заполняется минимально и записывается в ADR-0004, а не
выводится заново на каждой таблице:

- тип первичного ключа — `BIGSERIAL` везде (решение Архитектора, Issue #2);
- `PRIMARY KEY` у двенадцати таблиц, где написано просто `id`, — вывод из
  умолчания той же природы, что и тип;
- `created_at` — `TIMESTAMPTZ`, nullable и без server default, **кроме**
  `review_states.created_at`, где §3 прямо пишет `NOT NULL DEFAULT now()`.

Внешние ключи ставятся там и только там, где §3 пишет `FK`. В `llm_calls`
пометки `FK` нет ни у `user_id`, ни у `material_id`, поэтому ограничений там
нет — см. замечание в теле коммита: добавить их по своей инициативе значило бы
реализовать больше, чем требует §3.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    ARRAY,
    REAL,
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    Time,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Общий базовый класс. `Base.metadata` — вход для `alembic check`."""


def _pk() -> Mapped[int]:
    """Первичный ключ `BIGSERIAL` — ADR-0004."""
    return mapped_column(BigInteger, primary_key=True)


def _created_at() -> Mapped[datetime.datetime | None]:
    """`created_at` в форме §3: тип есть, обязательности и default нет.

    Значение проставляет приложение. Server default не добавляется: §3 его
    называет только у `review_states.created_at`, и дописывать остальным
    значило бы реализовать больше, чем требует спецификация.
    """
    return mapped_column(DateTime(timezone=True), nullable=True)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = _pk()
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    tz: Mapped[str | None] = mapped_column(Text, server_default="Europe/Moscow")
    quiet_from: Mapped[datetime.time | None] = mapped_column(Time, server_default="23:00")
    quiet_to: Mapped[datetime.time | None] = mapped_column(Time, server_default="08:00")
    daily_cap: Mapped[int | None] = mapped_column(Integer, server_default="60")
    new_per_day: Mapped[int | None] = mapped_column(Integer, server_default="20")
    session_size: Mapped[int | None] = mapped_column(Integer, server_default="6")
    lang: Mapped[str | None] = mapped_column(Text, server_default="ru")
    paused: Mapped[bool | None] = mapped_column(Boolean, server_default="false")
    plan: Mapped[str | None] = mapped_column(Text, server_default="free")
    """Задел под тарифы, в MVP не используется (§3)."""
    consent_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """Согласие на обработку данных. ТЗ §31: фиксируется при первом запуске."""
    created_at: Mapped[datetime.datetime | None] = _created_at()


class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = _pk()
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = _created_at()


class Material(Base):
    __tablename__ = "materials"
    __table_args__ = (
        # §3.1, дубликаты материала у одного пользователя. Частичный по
        # наличию ключа: §4.4 опознаёт дубль по sha256, а у материала без
        # него (например, ссылки) ограничения быть не должно.
        Index(
            "uq_materials_user_sha256",
            "user_id",
            text("(origin->>'sha256')"),
            unique=True,
            postgresql_where=text("origin ? 'sha256'"),
        ),
        # §3.1, задел под общий кеш обработки: один и тот же файл у разных
        # пользователей обрабатывается один раз.
        Index("ix_materials_sha256", text("(origin->>'sha256')")),
    )

    id: Mapped[int] = _pk()
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    subject_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("subjects.id"), nullable=True
    )
    kind: Mapped[str | None] = mapped_column(Text)
    """pdf|image|audio|video|web|text"""
    title: Mapped[str | None] = mapped_column(Text)
    origin: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    """{sha256, orig_name, path, url, pages, duration, youtube_id}"""
    status: Mapped[str | None] = mapped_column(Text)
    """queued|processing|ready|failed (§4.1)"""
    mode: Mapped[str | None] = mapped_column(Text, server_default="full")
    """full|notes_only (§6.3)"""
    completed_stages: Mapped[list[str] | None] = mapped_column(ARRAY(Text), server_default="{}")
    """Пропущенная стадия пишется как 'questions:skipped' (§3)."""
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cost_estimate: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime | None] = _created_at()
    processed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[int] = _pk()
    material_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("materials.id"), nullable=False)
    path: Mapped[str | None] = mapped_column(Text)
    mime: Mapped[str | None] = mapped_column(Text)
    w: Mapped[int | None] = mapped_column(Integer)
    h: Mapped[int | None] = mapped_column(Integer)
    tg_file_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)


class Fragment(Base):
    __tablename__ = "fragments"
    __table_args__ = (Index("ix_fragments_material_ord", "material_id", "ord"),)

    id: Mapped[int] = _pk()
    material_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("materials.id"), nullable=False)
    ord: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[str | None] = mapped_column(Text)
    """text|transcript|slide|image"""
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    asset_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("media_assets.id"), nullable=True
    )
    locator: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    """§3.4. Обязателен: без локатора фрагмент не даёт перехода к источнику."""
    quality: Mapped[float | None] = mapped_column(REAL, server_default="1.0")
    """0..1. Тип REAL задан §3 и определяет тип расчёта final_confidence."""
    tokens: Mapped[int | None] = mapped_column(Integer)


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = _pk()
    material_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("materials.id"), nullable=False)
    ord: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text)
    summary_md: Mapped[str | None] = mapped_column(Text)
    fragment_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[int] = _pk()
    material_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("materials.id"), nullable=False)
    section_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sections.id"), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    """Одно проверяемое утверждение, одно предложение."""
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    """1-2 предложения, используются в разборе."""
    kind: Mapped[str | None] = mapped_column(Text)
    """definition|cause|difference|property|formula|example"""
    fragment_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    """Обязателен: И-1, факт без evidence недопустим (§7.3 проверка 1)."""
    importance: Mapped[int | None] = mapped_column(SmallInteger, server_default="2")
    """1..3, где 3 = максимальная важность (§3.2)."""
    difficulty: Mapped[int | None] = mapped_column(SmallInteger, server_default="2")
    """1..3, сложность ЗНАНИЯ, не задания (§10.3)."""
    confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    """0..1. Тип REAL задан §3, см. `quality`."""
    derived: Mapped[bool | None] = mapped_column(Boolean, server_default="false")
    suspended: Mapped[bool | None] = mapped_column(Boolean, server_default="false")
    generation_status: Mapped[str | None] = mapped_column(Text, server_default="pending")
    """pending|ok|single_format|unavailable (§3.3)"""
    norm_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = _created_at()


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        Index("ix_questions_fact_status", "fact_id", "status"),
        # §3.1: без этого условие `EXISTS (... status='valid')` в планировщике
        # (§16) превращается в последовательное сканирование.
        Index(
            "ix_questions_fact_valid",
            "fact_id",
            postgresql_where=text("status = 'valid'"),
        ),
        # §3.1, И-6. Индекс **обязан** быть частичным и ограниченным именно
        # `valid`. Без условия отклонённая строка занимает слот навсегда:
        # перегенерация падает на IntegrityError, и факт теряет этот формат до
        # конца жизни. Условие `status <> 'rejected'`, стоявшее в v3.4,
        # закрывало половину проблемы — под него попадали строки `draft`,
        # которые остаются после упавшего воркера.
        Index(
            "uq_questions_fact_type_direction_valid",
            "fact_id",
            "type",
            "direction",
            unique=True,
            postgresql_where=text("status = 'valid'"),
        ),
    )

    id: Mapped[int] = _pk()
    fact_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("facts.id"), nullable=False)
    type: Mapped[str | None] = mapped_column(Text)
    """mcq|open|match_text|match_image"""
    direction: Mapped[str | None] = mapped_column(Text, server_default="forward")
    """forward|reverse"""
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    answer: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    difficulty: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="2")
    """Сложность ЗАДАНИЯ, используется в §13.3.

    `NOT NULL DEFAULT 2` — правка v3.4, внесённая потому, что `difficulty <= cap`
    при `NULL` даёт `NULL`, и вопрос молча выпадает из выборки планировщика.
    Обе половины свойства проверяются контрактом схемы.
    """
    status: Mapped[str | None] = mapped_column(Text, server_default="draft")
    """draft|valid|rejected (§3.3)"""
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(Text)
    generation_attempt: Mapped[int | None] = mapped_column(Integer, server_default="1")
    last_shown_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    shown_count: Mapped[int | None] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime.datetime | None] = _created_at()


class ReviewState(Base):
    __tablename__ = "review_states"
    __table_args__ = (
        # §3.1, выборка планировщика: что пора повторять, без приостановленных.
        Index(
            "ix_review_states_user_due",
            "user_id",
            "due_at",
            postgresql_where=text("NOT suspended"),
        ),
    )

    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), primary_key=True)
    fact_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("facts.id"), primary_key=True)
    """Составной первичный ключ `(user_id, fact_id)` — §3.

    Состояние повторения висит на факте, а не на вопросе (§0.2): иначе три
    формулировки одного знания получат три расписания.
    """
    due_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stability: Mapped[float | None] = mapped_column(Double)
    difficulty: Mapped[float | None] = mapped_column(Double)
    """§3 объявляет оба `DOUBLE PRECISION` — в отличие от `confidence` и
    `quality`, которые `REAL`. Это не разнобой: FSRS считает в float64, а
    `final_confidence` — произведение двух полей хранения."""
    state: Mapped[int | None] = mapped_column(SmallInteger)
    reps: Mapped[int | None] = mapped_column(Integer, server_default="0")
    lapses: Mapped[int | None] = mapped_column(Integer, server_default="0")
    last_review: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_q_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_direction: Mapped[str | None] = mapped_column(Text, nullable=True)
    seen: Mapped[bool | None] = mapped_column(Boolean, server_default="false")
    """§14.2: выставляется только после ответа или явного «Не знаю»."""
    suspended: Mapped[bool | None] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    introduced_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """Момент первого ответа, §16.1. Проставляется, если ещё пуст (§14.2)."""


class ReviewLog(Base):
    __tablename__ = "review_logs"

    id: Mapped[int] = _pk()
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    fact_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("facts.id"), nullable=False)
    question_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("questions.id"), nullable=False)
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    verdict: Mapped[str | None] = mapped_column(Text)
    """correct|partial|wrong|skipped"""
    scheduled_days: Mapped[int | None] = mapped_column(Integer)
    elapsed_days: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    fsrs_version: Mapped[str] = mapped_column(Text, nullable=False)
    planner_version: Mapped[str] = mapped_column(Text, nullable=False)
    """Обе версии обязательны: по §13.1 именно они делают возможным пересчёт
    расписания при смене параметров (отложен на этап 3, WP-26)."""
    reviewed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        # §3.1 и §38 №2: двух активных сессий одного пользователя быть не может.
        # Ограничение в БД, а не в коде: гонка двух воркеров проверкой в коде
        # не закрывается.
        Index(
            "uq_sessions_user_active",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = _pk()
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    kind: Mapped[str | None] = mapped_column(Text)
    """scheduled|manual|quick|exam"""
    material_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("materials.id"), nullable=True
    )
    status: Mapped[str | None] = mapped_column(Text)
    """active|finished|abandoned"""
    control_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """Сообщение с клавиатурой текущего задания (§20)."""
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class SessionItem(Base):
    __tablename__ = "session_items"

    id: Mapped[int] = _pk()
    session_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sessions.id"), nullable=False)
    ord: Mapped[int | None] = mapped_column(Integer)
    question_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("questions.id"), nullable=False)
    fact_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("facts.id"), nullable=False)
    user_state: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    """Промежуточный ввод match и `media_message_ids` (§20)."""
    answer_raw: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    rating: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class QualityReport(Base):
    __tablename__ = "quality_reports"

    id: Mapped[int] = _pk()
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    question_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("questions.id"), nullable=True
    )
    fact_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("facts.id"), nullable=True)
    section_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("sections.id"), nullable=True
    )
    kind: Mapped[str | None] = mapped_column(Text)
    """unclear|no_evidence|wrong_answer|wrong_note|duplicate|wrong_difficulty|technical"""
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(Text)
    """new|applied|dismissed"""
    created_at: Mapped[datetime.datetime | None] = _created_at()


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = _pk()
    user_id: Mapped[int | None] = mapped_column(BigInteger)
    material_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """Без внешнего ключа: §3 не помечает `FK` ни здесь, ни у `user_id`.

    Добавить ограничение по своей инициативе значило бы реализовать больше,
    чем требует спецификация (§40). Вынесено вопросом в ревью кода.
    """
    purpose: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    pricing_version: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_currency: Mapped[str | None] = mapped_column(Text)
    estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    actual_cost: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    ok: Mapped[bool | None] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime | None] = _created_at()
