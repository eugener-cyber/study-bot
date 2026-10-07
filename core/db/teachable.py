"""Предикат обучаемости — единственная точка обращения.

ТЗ §14.3: определение обучаемого факта используется в трёх местах — знаменатель
coverage (§14.1), фильтр планировщика (§16), создание `review_states` (§4.6) —
и реализуется **одним** SQL-предикатом, а не тремя похожими условиями в разных
модулях. Расхождение между ними даёт факты, которые числятся в статистике, но
никогда не выдаются, либо наоборот.

Здесь порог подставляется из настроек. В DDL функции его нет сознательно:
преамбула §30.3 требует, чтобы `MIN_FACT_CONFIDENCE` оставался константой
конфигурации, а не литералом. Поэтому все потребители обязаны ходить через
`teachable_facts_select`, а не вызывать SQL-функцию напрямую — иначе порог
снова станет литералом, только уже в коде потребителя. Это проверяет
`test_threshold_not_inlined_by_consumers`.
"""

from __future__ import annotations

from sqlalchemy import REAL, BigInteger, Select, bindparam, select, text
from sqlalchemy.sql.expression import TextualSelect

FUNCTION_NAME = "teachable_facts"

_CALL = text(
    # CAST(:min_confidence AS real) — приведение обязательно и явно.
    # Приведение float8 -> real в Postgres имеет уровень assignment, не
    # implicit, поэтому для разрешения функции его не хватает: вызов
    # `teachable_facts(0.65::double precision)` падает «function does not
    # exist». А `MIN_FACT_CONFIDENCE` приходит из `Settings` как Python float,
    # то есть float8. Без CAST запрос упал бы при первом же обращении.
    #
    # Если ошибка привязки когда-нибудь всплывёт снова — лечить приведением,
    # а не возвратом подписи к `double precision`: возврат снимет ошибку и
    # вернёт дефект, при котором факт ровно на пороге молча перестаёт быть
    # обучаемым на порогах 0.65, 0.7 и 0.9.
    f"SELECT fact_id, material_id, section_id, final_confidence "
    f"FROM {FUNCTION_NAME}(CAST(:min_confidence AS real))"
).columns(
    # Типы повторяют `RETURNS TABLE` функции. Повторение нежелательно, но
    # альтернатива — не объявлять их вовсе, и тогда SQLAlchemy отдаёт колонки
    # без типа, а `final_confidence` приходит потребителям как строка.
    # Расхождение с миграцией поймает `test_returned_confidence_is_never_null`:
    # он сравнивает значение с числом.
    fact_id=BigInteger,
    material_id=BigInteger,
    section_id=BigInteger,
    final_confidence=REAL,
)


def teachable_facts_select(min_confidence: float) -> TextualSelect:
    """Выборка обучаемых фактов при заданном пороге.

    Параметр передаётся привязкой, а не подстановкой в текст: подстановка
    открыла бы SQL-инъекцию через значение конфигурации и мешала бы Postgres
    переиспользовать план запроса.
    """
    return _CALL.bindparams(bindparam("min_confidence", value=min_confidence))


def teachable_fact_ids(min_confidence: float) -> Select[tuple[int]]:
    """Только идентификаторы — для `IN`-подзапросов у потребителей.

    Отдельная функция, а не `.with_only_columns()` на стороне вызывающего:
    иначе каждый потребитель собирал бы подзапрос сам, и «одна точка
    композиции» снова стала бы словесным правилом.
    """
    subquery = teachable_facts_select(min_confidence).subquery()
    return select(subquery.c.fact_id)
