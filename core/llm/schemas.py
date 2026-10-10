"""Схемы ответов модели. ТЗ §6.1, §7.1, §9.2, §10.1.

§6.1 требует структурированного вывода «со строгой схемой», и строгость здесь
буквальная: `extra="forbid"`. Лишнее поле в ответе означает, что модель поняла
задачу иначе, чем её ставили, и молча отбросить такое поле — худший из
вариантов: в §6.1 для ошибки схемы предусмотрен повтор с приклеенным текстом
ошибки, то есть механизм исправления уже есть, и он работает только если
расхождение считается ошибкой.

**Чего здесь нет и почему.**

*Проверок §11.1* — границы `correct_index`, биекция пар, дубли в `left`/`right`
— нет. Они принадлежат WP-10, который реализует §11 целиком и отклоняет
**отдельный вопрос**, сохраняя остальные (§11). Сделай их здесь — они стали бы
ошибкой схемы, то есть отвергали бы **весь набор** и расходовали повтор §6.1
на то, для чего §11 предусматривает другую реакцию. Здесь остаётся форма:
поля, типы и количественные пределы §9.2, которые §9.2 называет ограничениями
платформы, а не проверкой качества.

*Схемы конспекта* — нет, и это решение, а не пропуск. Шаблон §8 ссылается на
`section.summary_md` как на **вход**, а остальные четыре блока выводит из
сохранённых фактов: «Ключевые понятия» — факты с `kind = definition`,
«Основные положения» — факты секции, «Что запомнить» — факты с
`importance = 3`. Отдельной таблицы под конспект в §3 нет, и §8.1 отдаёт его
с пагинацией по требованию. Значит конспект **рендерится**, а не генерируется,
и схемы ответа у него быть не может.

Это расходится с §6.3, где `notes` учитывается как стадия с расходом токенов
(коэффициент `notes_output_ratio`). Расхождение вынесено change request'ом —
решать его самостоятельно §40 запрещает, а до решения оценка остаётся
завышенной, что безопаснее заниженной.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

FactKind = Literal["definition", "cause", "difference", "property", "formula", "example"]
"""§7.1 и §3: `kind` факта. Literal, а не строка: неизвестный вид приехал бы в
базу и выпал бы из подсчётов §3.2, не нарушив ни одного ограничения."""

QuestionType = Literal["mcq", "open", "match_text", "match_image"]
"""§9.1. MVP — четыре типа; `cloze`, `ordering`, `true_false` добавляются
«без изменения модели», то есть расширением этого перечня."""

Direction = Literal["forward", "reverse"]
"""§10.1: `forward` — термин → определение, `reverse` — определение → термин."""

Grade = Annotated[int, Field(ge=1, le=3)]
"""Шкала важности и сложности §3.2: 3 — ключевой, 2 — существенный,
1 — вспомогательный. `SMALLINT` в §3 допускает любое число, но смысл имеют
только эти три."""

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""§7.1: `confidence` в пределах 0..1. Выход за пределы ломает §7.3, где
`final_confidence = confidence × min(quality)` сравнивается с порогом."""


class StrictAnswer(BaseModel):
    """Общая база: лишних полей нет, пропущенных полей нет.

    `extra="forbid"` — требование §6.1 «строгая схема». `populate_by_name`
    не включается: имена полей здесь ровно те, что в §7.1 и §9.2, и
    синонимов у них быть не должно — иначе промпт и схема начнут расходиться
    в названиях, а расхождение проявится как ошибка схемы без причины.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------- sections


class SectionPlan(StrictAnswer):
    """Одна секция: заголовок и фрагменты, которые в неё вошли.

    Порядкового номера нет: он берётся из позиции в списке. Номер от модели
    пришлось бы проверять на пропуски и дубли, и при конфликте всё равно
    побеждала бы позиция — §3 требует `ord` без дыр, потому что по нему
    идёт навигация §8.1.
    """

    title: str = Field(min_length=1)
    fragment_ids: list[int] = Field(min_length=1)


class SectionsAnswer(StrictAnswer):
    """Разбиение материала на секции (§7.1: «фрагменты группируются в секции»).

    Формы ответа ТЗ для этой стадии не задаёт — §7.1 называет только
    результат группировки, а семантическая нарезка по `MAX_SECTION_TOKENS`
    принадлежит §6.2 и WP-07. Поэтому схема минимальна: ровно те поля
    `sections` из §3, которые нельзя вывести из других данных.
    """

    sections: list[SectionPlan] = Field(min_length=1)


# ------------------------------------------------------------------ facts


class FactDraft(StrictAnswer):
    """Факт в ответе модели. Поля — буквально §7.1.

    Умолчаний нет ни у одного поля, хотя §3 даёт их в базе. Причина в том,
    что умолчание превращает **пропуск** в утверждение. Самое дорогое —
    `derived`: по §7.3 проверка 4 выведенный факт не участвует в SRS, и
    умолчание `false` отправило бы догадку модели в обучение как
    подтверждённое знание. У `importance` цена ниже, но та же природа:
    пропуск стал бы «существенный», и факт попал бы в средний режим §8.
    """

    statement: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    kind: FactKind
    fragment_ids: list[int] = Field(min_length=1)
    """§7.3 проверка 1: `fragment_ids ⊆ submitted_fragment_ids`, иначе факт
    отбрасывается. Само подмножество проверяет стадия — ей известен состав
    поданных фрагментов, схеме нет. Здесь только непустота: факт без
    evidence нарушает И-1 независимо от того, что подавали."""

    importance: Grade
    difficulty: Grade
    """Сложность **знания**, не задания (§10.3). Их смешение запрещено:
    простое знание допускает трудное задание и наоборот."""

    derived: bool
    confidence: Confidence


class FactsAnswer(StrictAnswer):
    """Ответ стадии `facts` — ровно структура из §7.1."""

    title: str = Field(min_length=1)
    summary_md: str = Field(min_length=1)
    """3–8 предложений (§7.1). Число предложений схемой не проверяется: это
    требование к промпту, а подсчёт предложений регуляркой отбраковывал бы
    правильные конспекты из-за сокращений и формул."""

    facts: list[FactDraft]
    """Пустой список допустим: секция без проверяемых утверждений бывает —
    оглавление, список литературы, титул. Отвергать такой ответ значило бы
    требовать от модели выдумать факты там, где их нет."""


# -------------------------------------------------------------- questions


class McqPayload(StrictAnswer):
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=4)
    """§9.2: «вариантов mcq не более 4» — ограничение платформы, не качества:
    на кнопках остаются цифры, а текст уходит в тело сообщения."""


class McqAnswer(StrictAnswer):
    correct_index: int = Field(ge=0)
    """Верхняя граница не проверяется здесь: §11.1 относит «`correct_index` в
    границах массива» к валидации, которая отклоняет один вопрос, а не набор."""


class McqQuestion(StrictAnswer):
    type: Literal["mcq"]
    direction: Direction
    difficulty: Grade
    payload: McqPayload
    answer: McqAnswer


class OpenPayload(StrictAnswer):
    question: str = Field(min_length=1)
    hint: str | None = None
    """§9.2 допускает `null`. Умолчание здесь уместно, в отличие от полей
    факта: отсутствие подсказки — это и есть «подсказки нет», а не пропуск
    утверждения о факте."""


class OpenAnswer(StrictAnswer):
    reference: str = Field(min_length=1)
    accept: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    """§9.2: «подсказка судье, не автовердикт». Поле не делает вердикт сам —
    иначе ответ своими словами без ключевого слова считался бы неверным."""


class OpenQuestion(StrictAnswer):
    type: Literal["open"]
    direction: Direction
    difficulty: Grade
    payload: OpenPayload
    answer: OpenAnswer


class MatchAnswer(StrictAnswer):
    pairs: list[tuple[int, int]] = Field(min_length=2)
    """Биекцию пар проверяет §11.1 (WP-10), здесь только форма."""


class MatchTextPayload(StrictAnswer):
    prompt: str = Field(min_length=1)
    left: list[str] = Field(min_length=3, max_length=5)
    right: list[str] = Field(min_length=3, max_length=5)
    """§9.2: текстовое соотнесение 3–5 пар."""


class MatchTextQuestion(StrictAnswer):
    type: Literal["match_text"]
    direction: Direction
    difficulty: Grade
    payload: MatchTextPayload
    answer: MatchAnswer


class MatchImagePayload(StrictAnswer):
    prompt: str = Field(min_length=1)
    left: list[str] = Field(min_length=2, max_length=4)
    asset_ids: list[int] = Field(min_length=2, max_length=4)
    """§9.2: картиночное соотнесение 2–4 пары. `asset_ids` ссылаются на
    `media_assets` (§3); §10.2 №4 допускает этот тип только при наличии
    подходящих ассетов, и проверяет это стадия — схеме состав ассетов
    материала неизвестен."""


class MatchImageQuestion(StrictAnswer):
    type: Literal["match_image"]
    direction: Direction
    difficulty: Grade
    payload: MatchImagePayload
    answer: MatchAnswer


QuestionDraft = McqQuestion | OpenQuestion | MatchTextQuestion | MatchImageQuestion
"""Один вопрос любого типа §9.1, без дискриминатора.

Нужен потребителям как обычный тип: `AnyQuestion` обёрнут в `Annotated` с
`Field`, и в подписи функции такая обёртка читается как поле модели, хотя
означает ровно то же множество классов.
"""

AnyQuestion = Annotated[
    QuestionDraft,
    Field(discriminator="type"),
]
"""Дискриминированный union по `type` — §9.1 называет его прямо.

Дискриминатор существен для сообщения об ошибке: без него pydantic
перечисляет ошибки всех четырёх вариантов, и приклеенный к повтору текст
(§6.1) становится нечитаемым — модель получает четыре противоречащих
замечания вместо одного.
"""


class QuestionSet(StrictAnswer):
    """Набор вопросов одного факта. §10.1: один вызов на факт.

    `2..4` — предел §10.1 и §10.2 №1. Остальные правила набора — минимум два
    разных типа, уникальность `(type, direction)`, наличие `open` по CR-1б —
    схемой не выражены: §10.2 предписывает им **одну повторную генерацию**, а
    затем `generation_status = 'single_format'` с сохранением тех вопросов,
    которые получились. Ошибка схемы по §6.1 ведёт себя иначе — повтор, затем
    `failed`, — и выразить одно другим значило бы потерять уже годные вопросы.
    """

    questions: list[AnyQuestion] = Field(min_length=2, max_length=4)


SCHEMA_BY_PURPOSE: dict[str, type[BaseModel]] = {
    "sections": SectionsAnswer,
    "facts": FactsAnswer,
    "questions": QuestionSet,
}
"""Назначение вызова → схема ответа.

`notes` отсутствует намеренно — см. докстроку модуля. Словарь нужен не
клиенту (тот получает схему параметром), а проверкам: он даёт перечислить все
назначения и убедиться, что у каждого есть и схема, и коэффициент выхода в
`StageCoefficients`.
"""
