"""Схема в живой базе сверяется с §3 и §3.1.

`test_models.py` проверяет объявления, этот файл — то, что в базе реально
создалось миграциями. Разница существенная: модель можно поправить, миграцию
забыть, и первый набор тестов останется зелёным.

Индексы проверяются **по определению, а не по счёту строк**. `pg_indexes`
перечисляет и индексы под первичные ключи, и под `UNIQUE`-ограничения, и
служебный индекс таблицы `alembic_version`: на нашей схеме это 24 строки, а не
8. Ассерт на число упал бы, а подобранный под него фильтр был бы подгонкой под
тест — прямой урок `>= 26` из WP-01.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from core.db.models import Base

pytestmark = pytest.mark.db

ALEMBIC_TABLE = "alembic_version"
"""Служебная таблица Alembic. В §3 её нет и быть не должно: она не часть
модели данных, а журнал применённых миграций."""

# §3.1 целиком. Для каждого индекса — фрагменты, которые обязаны быть в
# `pg_indexes.indexdef`. У частичных уникальных проверяются ОБЕ половины:
# проверка только условия `WHERE` прошла бы и на неуникальном индексе.
EXPECTED_INDEXES: dict[str, tuple[str, ...]] = {
    "uq_materials_user_sha256": (
        "CREATE UNIQUE INDEX",
        "user_id",
        "origin ->> 'sha256'",
        "WHERE (origin ? 'sha256'",
    ),
    "ix_materials_sha256": ("CREATE INDEX", "origin ->> 'sha256'"),
    "ix_fragments_material_ord": ("CREATE INDEX", "material_id, ord"),
    "ix_review_states_user_due": (
        "CREATE INDEX",
        "user_id, due_at",
        "WHERE (NOT suspended)",
    ),
    "uq_sessions_user_active": (
        "CREATE UNIQUE INDEX",
        "user_id",
        "WHERE (status = 'active'",
    ),
    "ix_questions_fact_status": ("CREATE INDEX", "fact_id, status"),
    "ix_questions_fact_valid": (
        "CREATE INDEX",
        "fact_id",
        "WHERE (status = 'valid'",
    ),
    "uq_questions_fact_type_direction_valid": (
        "CREATE UNIQUE INDEX",
        "fact_id, type, direction",
        "WHERE (status = 'valid'",
    ),
}


async def _rows(engine: AsyncEngine, sql: str) -> list[tuple[object, ...]]:
    async with engine.connect() as connection:
        result = await connection.execute(text(sql))
        return [tuple(row) for row in result]


async def test_all_spec_tables_exist(engine: AsyncEngine) -> None:
    """В базе ровно таблицы §3 плюс служебная таблица Alembic."""
    rows = await _rows(
        engine,
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'",
    )
    actual = {str(name) for (name,) in rows} - {ALEMBIC_TABLE}
    assert actual == set(Base.metadata.tables)


@pytest.mark.parametrize("name", sorted(EXPECTED_INDEXES))
async def test_index_definition_matches_spec(engine: AsyncEngine, name: str) -> None:
    """Каждый индекс §3.1 существует и определён так, как требует §3.1."""
    rows = await _rows(
        engine,
        f"SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND indexname='{name}'",
    )
    assert rows, f"индекса {name} нет в базе"
    definition = str(rows[0][0])
    for fragment in EXPECTED_INDEXES[name]:
        assert fragment in definition, f"{name}: в определении нет «{fragment}»\n{definition}"


async def test_slot_index_is_unique_and_partial(engine: AsyncEngine) -> None:
    """И-6 отдельным тестом: индекс слота уникален и ограничен именно `valid`.

    Выделено из общей параметризации, потому что это единственный индекс, у
    которого потеря любой из двух половин даёт аварию, а не просадку скорости.
    Без `UNIQUE` нарушается §38 №5. Без `WHERE status='valid'` отклонённая
    строка занимает слот навсегда: перегенерация падает на `IntegrityError`,
    и факт теряет формат вопроса до конца жизни.
    """
    rows = await _rows(
        engine,
        "SELECT indexdef FROM pg_indexes WHERE schemaname='public' "
        "AND indexname='uq_questions_fact_type_direction_valid'",
    )
    definition = str(rows[0][0])
    assert definition.startswith("CREATE UNIQUE INDEX"), definition
    assert "WHERE (status = 'valid'::text)" in definition, definition
    assert "rejected" not in definition, "условие v3.4 `status <> 'rejected'` вернулось"


async def test_primary_key_indexes_cover_every_table(engine: AsyncEngine) -> None:
    """Первичный ключ есть у каждой таблицы §3 — четырнадцать индексов.

    Число не задано литералом: он берётся из состава таблиц. Четырнадцать, а
    не тринадцать, потому что составной PK `review_states` — тоже один индекс.
    """
    rows = await _rows(
        engine,
        "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_class t ON t.oid = i.indrelid "
        "JOIN pg_namespace n ON n.oid = t.relnamespace "
        "WHERE i.indisprimary AND n.nspname='public'",
    )
    names = {str(name) for (name,) in rows}
    expected = {f"{table}_pkey" for table in Base.metadata.tables}
    assert expected <= names, f"нет первичного ключа: {sorted(expected - names)}"
    assert len(expected) == len(Base.metadata.tables)


async def test_id_columns_are_bigint(engine: AsyncEngine) -> None:
    """`BIGSERIAL` везде — решение Архитектора, ADR-0004."""
    rows = await _rows(
        engine,
        "SELECT table_name, data_type FROM information_schema.columns "
        "WHERE table_schema='public' AND column_name='id'",
    )
    wrong = {str(t): str(d) for t, d in rows if str(d) != "bigint"}
    assert not wrong, f"не bigint: {wrong}"


async def test_not_null_matches_models(engine: AsyncEngine) -> None:
    """Обязательность колонок в базе совпадает с моделями.

    Сверяется весь состав, а не тринадцать явных пометок §3: модель могла
    объявить колонку обязательной, а миграция — нет, и `test_models.py` этого
    не увидит.
    """
    rows = await _rows(
        engine,
        "SELECT table_name, column_name, is_nullable FROM information_schema.columns "
        f"WHERE table_schema='public' AND table_name <> '{ALEMBIC_TABLE}'",
    )
    mismatched: list[str] = []
    for table, column, is_nullable in rows:
        declared = Base.metadata.tables[str(table)].columns[str(column)].nullable
        actual = str(is_nullable) == "YES"
        if declared is not actual:
            mismatched.append(f"{table}.{column}: модель nullable={declared}, база {actual}")
    assert not mismatched, "\n".join(mismatched)


async def test_column_defaults_are_present(engine: AsyncEngine) -> None:
    """Каждый `server_default` модели существует и в базе.

    `alembic check` сравнивает default только при `compare_server_default=True`
    (выставлен явно в `env.py`), а дефолт этого параметра — `False`. Проверка
    по `information_schema` не зависит от конфигурации Alembic вовсе.
    """
    rows = await _rows(
        engine,
        "SELECT table_name, column_name, column_default FROM information_schema.columns "
        f"WHERE table_schema='public' AND table_name <> '{ALEMBIC_TABLE}'",
    )
    actual = {(str(t), str(c)): d for t, c, d in rows}
    missing: list[str] = []
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if column.server_default is None or column.primary_key:
                continue
            if actual.get((table.name, column.name)) is None:
                missing.append(f"{table.name}.{column.name}")
    assert not missing, f"server_default объявлен в модели, но отсутствует в базе: {missing}"


async def test_question_difficulty_default_is_two(engine: AsyncEngine) -> None:
    """`questions.difficulty NOT NULL DEFAULT 2` — обе половины, в базе.

    Правка v3.4: при `NULL` условие `difficulty <= cap` даёт `NULL`, и вопрос
    молча выпадает из выборки планировщика (§13.3).
    """
    rows = await _rows(
        engine,
        "SELECT is_nullable, column_default FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='questions' AND column_name='difficulty'",
    )
    is_nullable, default = rows[0]
    assert str(is_nullable) == "NO"
    assert default is not None and "2" in str(default)


async def test_confidence_and_quality_are_real_in_database(engine: AsyncEngine) -> None:
    """Оба множителя `final_confidence` хранятся в `real`.

    От этого зависит тип параметра `teachable_facts(min_confidence real)`:
    сравнение в `double precision` даёт ложь на порогах 0.65, 0.7 и 0.9.
    """
    rows = await _rows(
        engine,
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='public' AND ((table_name='fragments' AND column_name='quality') "
        "OR (table_name='facts' AND column_name='confidence'))",
    )
    assert len(rows) == 2
    for table, column, data_type in rows:
        assert str(data_type) == "real", f"{table}.{column} = {data_type}"
