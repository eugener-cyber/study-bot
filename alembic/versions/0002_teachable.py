"""teachable

Функция предиката обучаемости (ТЗ §14.3).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05

Миграция написана руками, а не автогенерацией: Alembic функции не отслеживает
вовсе. Отсюда следствие, которое надо знать: `alembic check` расхождение этой
функции с ожиданиями **не поймает** — её сторожат только тесты
`test_teachable.py`.

**Почему функция, а не view.** View, отфильтрованный по порогу, означал бы
`0.55` литералом здесь, в DDL. Преамбула §30.3: «Все значения ниже — константы
конфигурации, а не литералы в коде. Это ровно те числа, которые придётся
крутить после первых материалов». Иначе оператор правит порог в `.env`, он
меняется для генерации вопросов (§7.3) и не меняется для coverage (§14.1) и
планировщика (§16) — расхождение молчаливое.

**Почему параметр `real`, а не `double precision`.** §3 объявляет оба
множителя `final_confidence` как `REAL`. Произведение считается в float4, и
повышение до float8 ради сравнения вскрывает ошибку представления: факт ровно
на пороге перестаёт быть обучаемым. Прогон по значениям порога показал ложь
на 0.65, 0.7 и 0.9 — а дефолт 0.55 в ложь не попадает, поэтому тест на одном
дефолте был бы зелёным по совпадению.

**`LANGUAGE sql` и `STABLE`** выбраны, чтобы планировщик встраивал тело в
запрос: один вызов на запрос, а не на строку.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# COALESCE в УСЛОВИИ отбора стоять не должен, и это выяснилось мутационным
# прогоном. В первой версии функции он там был, с обоснованием «факт без
# evidence отсекает порог». Прогон показал, что обоснование верно только для
# положительного порога: при `MIN_FACT_CONFIDENCE = 0` произведение давало
# `0 >= 0`, и факт без evidence становился обучаемым — прямое нарушение И-1
# (§7.3 проверка 1). Ноль оператор поставить может: §30.3 прямо называет этот
# порог одним из тех, которые придётся крутить.
#
# Поэтому в `WHERE` произведение считается без COALESCE: `MIN` по пустому
# множеству, по несуществующим идентификаторам или по `NULL` в `quality` даёт
# `NULL`, и строка отбрасывается при любом пороге. В списке колонок COALESCE
# остаётся: он гарантирует, что `final_confidence` не придёт потребителям
# как `NULL`.
#
# COALESCE на suspended, derived и generation_status — другой случай и нужен:
# §3 объявляет их nullable с DEFAULT, то есть NULL достижим при вставке с
# явным NULL, и без COALESCE условие дало бы NULL, а факт выпал бы молча.
TEACHABLE = """
CREATE OR REPLACE FUNCTION teachable_facts(min_confidence real)
RETURNS TABLE (
    fact_id bigint,
    material_id bigint,
    section_id bigint,
    final_confidence real
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        f.id,
        f.material_id,
        f.section_id,
        (f.confidence * COALESCE(
            (SELECT MIN(fr.quality) FROM fragments fr WHERE fr.id = ANY(f.fragment_ids)),
            0::real
        ))::real
    FROM facts f
    WHERE NOT COALESCE(f.suspended, false)
      AND NOT COALESCE(f.derived, false)
      AND COALESCE(f.generation_status, 'pending') <> 'unavailable'
      AND (f.confidence
           * (SELECT MIN(fr.quality) FROM fragments fr WHERE fr.id = ANY(f.fragment_ids))
          ) >= min_confidence
      AND EXISTS (
            SELECT 1 FROM questions q
            WHERE q.fact_id = f.id AND q.status = 'valid'
      )
$$;
"""


def upgrade() -> None:
    op.execute(TEACHABLE)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS teachable_facts(real);")
