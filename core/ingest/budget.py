"""Оценка стоимости обработки и гейт. ТЗ §6.3.

**Оценка считается до стадии `extract`, а не после неё.** §6.3 объясняет
причину: для сканированного PDF `extract` — это vision-вызов на каждую
страницу, то есть основная статья расхода по материалу, и гейт после него
спрашивал бы «обрабатывать?» после того, как деньги уже потрачены. На
трёхсотстраничном скане такой гейт бессмыслен.

Оценка строится в два приёма:

```
probe    -> pages | duration | размер текста          -> предварительная, гейт
extract  -> fragments × average_size × coefficients   -> уточнённая
```

Предварительная сохраняется в `materials.cost_estimate` и служит основанием
гейта. После `extract` она пересчитывается по фактическому числу фрагментов и
перезаписывается там же; именно уточнённая сравнивается с фактом в §6.4.

**Про числа.** §6.3 называет «коэффициенты стадий», но нигде их не задаёт, а
`COST_WARN_THRESHOLD` в §30.3 стоит как `TODO(owner)` и выводится из спайка
(§34.1), который не выполнялся — ключа провайдера нет. Поэтому:

- коэффициенты стадий и цены за токен живут в `core/llm/estimation.py`, а не
  здесь. В WP-03 они были здесь, и это было неверным направлением
  зависимости: их второй потребитель — клиент LLM (§6.4, оценка и факт по
  каждому вызову), и он тянул бы за собой весь пакет обработки материала.
  Направление `ingest → llm` правильное: оценка стоимости материала законно
  зависит от цен и объёмов вызовов модели, а не наоборот;
- цена берётся по **провайдеру из настроек**, а не по `anthropic` жёстко: в
  режиме `manual` вызовов нет, цена нулевая (§2.1), и оценка в деньгах
  честно обращается в ноль, тогда как оценка в токенах остаётся осмысленной;
- пока `COST_WARN_THRESHOLD` не задан, гейт **не срабатывает**, и это
  записано явно: молчаливое срабатывание на `None` остановило бы обработку
  любого материала.

Расхождение вынесено отдельным change request — спецификация не задаёт
коэффициенты, и выдумывать их молча нельзя (§40).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from core.adapters.base import SourceMeta
from core.config import Settings
from core.llm.estimation import ProviderPrices, StageCoefficients, prices_for
from core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CostEstimate:
    """То, что попадает в `materials.cost_estimate` (§3).

    Состав полей задан §3: `{fragments, est_tokens, est_cost, currency}`.
    """

    fragments: int
    est_tokens: int
    est_cost: Decimal
    currency: str
    coefficients: StageCoefficients = field(default_factory=StageCoefficients)

    def as_json(self) -> dict[str, object]:
        """Для записи в JSONB. `Decimal` не сериализуется — отдаём строкой.

        Строкой, а не `float`: §6.4 сравнивает оценку с фактическим расходом,
        и потеря точности при двойном преобразовании давала бы расхождение
        там, где его нет.
        """
        return {
            "fragments": self.fragments,
            "est_tokens": self.est_tokens,
            "est_cost": str(self.est_cost),
            "currency": self.currency,
        }


def estimate_from_probe(
    meta: SourceMeta,
    settings: Settings,
    *,
    coefficients: StageCoefficients | None = None,
    prices: ProviderPrices | None = None,
) -> CostEstimate:
    """Предварительная оценка по результату `probe`. Основание гейта (§6.3).

    `fragments` здесь ещё неизвестно и остаётся нулём: поле заполняется
    уточнённой оценкой после `extract`. Ноль честнее выдуманного числа —
    по нему видно, что оценка предварительная.
    """
    coefficients = coefficients or StageCoefficients()
    prices = prices or prices_for(settings.LLM_PROVIDER)

    pages = meta.pages or 0
    vision_pages = round(pages * meta.vision_pages_ratio)
    text_pages = max(pages - vision_pages, 0)

    # Распознанный текст сканированных страниц читают содержательные стадии
    # так же, как текст обычных. Первая версия расчёта этого не учитывала, и
    # полностью сканированный материал выходил ДЕШЕВЛЕ текстового: vision
    # оплачивался, а четыре стадии после него получали ноль токенов на входе.
    # Поймал тест `test_scanned_pages_cost_more_than_text_pages`.
    source_tokens = (
        text_pages * coefficients.text_tokens_per_page
        + vision_pages * coefficients.vision_output_per_page
    )
    if meta.duration_sec:
        source_tokens += int(meta.duration_sec / 60 * coefficients.audio_tokens_per_minute)
    if meta.text_chars and not pages:
        # Источник без страниц и без дорожки: текст или веб-страница.
        source_tokens += meta.text_chars // 5

    input_tokens = vision_pages * coefficients.vision_input_per_page
    output_tokens = vision_pages * coefficients.vision_output_per_page

    # Содержательные стадии читают весь материал и пишут долю от него.
    # Их три, а не четыре: конспект §8 собирается из фактов без вызова
    # модели (решение Архитектора v3.13, CR-O).
    for ratio in (
        coefficients.sections_output_ratio,
        coefficients.facts_output_ratio,
        coefficients.questions_output_ratio,
    ):
        input_tokens += source_tokens
        output_tokens += int(source_tokens * ratio)

    return CostEstimate(
        fragments=0,
        est_tokens=input_tokens + output_tokens,
        est_cost=prices.cost(input_tokens, output_tokens),
        currency=settings.COST_CURRENCY,
        coefficients=coefficients,
    )


def estimate_after_extract(
    fragments: int,
    fragment_tokens: int,
    settings: Settings,
    *,
    coefficients: StageCoefficients | None = None,
    prices: ProviderPrices | None = None,
) -> CostEstimate:
    """Уточнённая оценка по фактическому числу фрагментов (§6.3).

    Vision уже оплачен на стадии `extract`, поэтому в уточнённую оценку входят
    только содержательные стадии. Именно эта оценка сравнивается с фактом в
    §6.4 — сравнивать предварительную бессмысленно, она считалась до того, как
    стало известно, сколько текста в материале на самом деле.
    """
    coefficients = coefficients or StageCoefficients()
    prices = prices or prices_for(settings.LLM_PROVIDER)

    input_tokens = 0
    output_tokens = 0
    for ratio in (
        coefficients.sections_output_ratio,
        coefficients.facts_output_ratio,
        coefficients.questions_output_ratio,
    ):
        input_tokens += fragment_tokens
        output_tokens += int(fragment_tokens * ratio)

    return CostEstimate(
        fragments=fragments,
        est_tokens=input_tokens + output_tokens,
        est_cost=prices.cost(input_tokens, output_tokens),
        currency=settings.COST_CURRENCY,
        coefficients=coefficients,
    )


def exceeds_threshold(estimate: CostEstimate, settings: Settings) -> bool:
    """Превышает ли оценка `COST_WARN_THRESHOLD` (§6.3).

    При незаданном пороге возвращает `False`, и это решение, а не недосмотр.
    `COST_WARN_THRESHOLD` в §30.3 стоит как `TODO(owner)` и выводится из
    спайка (§34.1), который не выполнялся. Срабатывание на `None` остановило
    бы обработку любого материала — то есть продукт не работал бы вовсе до
    получения ключа провайдера.
    """
    if settings.COST_WARN_THRESHOLD is None:
        log.info("cost_gate_disabled", reason="COST_WARN_THRESHOLD не задан")
        return False
    return estimate.est_cost > settings.COST_WARN_THRESHOLD


def needs_reestimate_confirmation(
    preliminary: CostEstimate, refined: CostEstimate, settings: Settings
) -> bool:
    """Выросла ли оценка больше, чем в `COST_REESTIMATE_FACTOR` раз (§6.3).

    Без этой проверки гейт обходится любым материалом, чью стоимость `probe`
    недооценил, — §6.3 говорит это прямо. Поэтому порог здесь не зависит от
    `COST_WARN_THRESHOLD` и работает, даже когда гейт выключен: вопрос задаётся
    по относительному росту, а не по абсолютной сумме.
    """
    if preliminary.est_cost <= 0:
        return refined.est_cost > 0
    factor = refined.est_cost / preliminary.est_cost
    return factor > Decimal(str(settings.COST_REESTIMATE_FACTOR))


def gate_question(estimate: CostEstimate, pages: int | None) -> str:
    """Текст вопроса пользователю. §6.3 задаёт его форму.

    Формулировка из §6.3 воспроизводится близко к букве: размер, оценка
    токенов, сумма с валютой. Цифры в вопросе — не украшение: без них выбор
    между «обработать полностью» и «только конспект» делается наугад.
    """
    size = f"~{pages} страниц" if pages else "неизвестный объём"
    return (
        f"Материал большой: {size}, примерно {estimate.est_tokens} токенов, "
        f"≈ {estimate.est_cost:.2f} {estimate.currency}."
    )
