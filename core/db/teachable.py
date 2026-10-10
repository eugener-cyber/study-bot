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


_QUALITY = "(SELECT MIN(fr.quality) FROM fragments fr WHERE fr.id = ANY(f.fragment_ids))"
"""Минимальное качество фрагментов факта — второй множитель §7.3.

Вынесено строкой, потому что встречается в выборке дважды: в выражении
`final_confidence` и в условии порога. Два текста одного подзапроса рано или
поздно разошлись бы, и условие стало бы отбирать по одному числу, а
возвращать другое.
"""

_GENERATABLE = text(
    "SELECT f.id AS fact_id, f.material_id, f.section_id, "
    f"(f.confidence * COALESCE({_QUALITY}, 0::real))::real AS final_confidence "
    "FROM facts f "
    "WHERE NOT COALESCE(f.suspended, false) "
    "  AND NOT COALESCE(f.derived, false) "
    f"  AND (f.confidence * {_QUALITY}) >= CAST(:min_confidence AS real)"
).columns(
    fact_id=BigInteger,
    material_id=BigInteger,
    section_id=BigInteger,
    final_confidence=REAL,
)


def generatable_facts_select(min_confidence: float) -> TextualSelect:
    """Факты, по которым **можно генерировать** вопросы. §7.3 проверки 3 и 4.

    Это не то же, что обучаемость, и разница существенна. `teachable_facts`
    требует наличия хотя бы одного вопроса со статусом `valid` — для
    планировщика и статистики это правильно: факт без вопросов выдать нельзя.
    Но стадия генерации вопросов работает **до** их появления, и предикат
    обучаемости на этом шаге пуст по построению: ни по одному факту вопросов
    ещё нет, и генерировать было бы не по чему.

    Поэтому здесь ровно две проверки §7.3: `final_confidence` не ниже порога
    (проверка 3 — «вопросы по нему не генерируются») и `derived = false`
    (проверка 4 — выведенные факты не участвуют в SRS, а вопрос существует
    только ради SRS). Остальные условия обучаемости к генерации не относятся.

    Связь двух предикатов держит тест, а не соглашение: разница между ними
    обязана быть ровно множеством фактов без валидных вопросов. Иначе при
    правке одного второй молча начнёт отбирать другое, и факты либо получат
    вопросы, но не попадут в расписание, либо наоборот.

    `generation_status` в условии не участвует намеренно. Значение
    `unavailable` означает, что вопросы сделать не удалось; запрещать повторную
    попытку значило бы делать отказ окончательным, тогда как §12.2 обещает
    перегенерацию, а §38 №19 — возврат факта в ротацию.
    """
    return _GENERATABLE.bindparams(bindparam("min_confidence", value=min_confidence))
