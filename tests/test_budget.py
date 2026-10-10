"""Оценка стоимости и гейт бюджета. ТЗ §6.3.

**Про `LLM_PROVIDER="anthropic"` в тестах стоимости.** Цены берутся по
провайдеру из настроек, а у `manual` они нулевые (§2.1): вызовов нет, платить
нечего. Поэтому любая проверка, сравнивающая **суммы**, на умолчании тестов
выродилась бы в `0 < 0` и прошла бы при полностью сломанном расчёте. Проверки
стоимости поэтому идут на платном провайдере, а вырожденность ручного режима
проверяется отдельно и называется прямо — см. `test_manual_provider_*`.
"""

from __future__ import annotations

from decimal import Decimal

from core.adapters.base import SourceMeta
from core.ingest.budget import (
    CostEstimate,
    estimate_after_extract,
    estimate_from_probe,
    exceeds_threshold,
    gate_question,
    needs_reestimate_confirmation,
)
from core.llm.estimation import StageCoefficients
from tests.test_dispatcher import _settings


def test_scanned_pages_cost_more_than_text_pages() -> None:
    """Vision — главный множитель стоимости (§5.2, §6.3).

    Это и есть причина, по которой гейт стоит до `extract`: сканированная
    страница требует vision-вызова, то есть основной статьи расхода, и
    спрашивать после неё поздно.
    """
    settings = _settings(LLM_PROVIDER="anthropic")
    text = estimate_from_probe(SourceMeta(pages=100, vision_pages_ratio=0.0), settings)
    scanned = estimate_from_probe(SourceMeta(pages=100, vision_pages_ratio=1.0), settings)

    assert scanned.est_cost > text.est_cost
    assert scanned.est_tokens > text.est_tokens


def test_preliminary_estimate_has_no_fragments() -> None:
    """`fragments` в предварительной оценке ноль, а не выдуманное число.

    §3 задаёт состав `cost_estimate`, и по нулю видно, что оценка
    предварительная. Выдуманное число нельзя было бы отличить от настоящего.
    """
    estimate = estimate_from_probe(SourceMeta(pages=10), _settings())
    assert estimate.fragments == 0


def test_estimate_grows_with_size() -> None:
    settings = _settings(LLM_PROVIDER="anthropic")
    small = estimate_from_probe(SourceMeta(pages=10), settings)
    large = estimate_from_probe(SourceMeta(pages=300), settings)
    assert large.est_cost > small.est_cost * 20


def test_audio_duration_is_counted() -> None:
    """Источник без страниц, но с дорожкой, оценивается по длительности."""
    settings = _settings()
    silence = estimate_from_probe(SourceMeta(), settings)
    hour = estimate_from_probe(SourceMeta(duration_sec=3600), settings)
    assert hour.est_tokens > silence.est_tokens


def test_plain_text_without_pages_is_counted_by_chars() -> None:
    """Текст и веб-страница не имеют ни страниц, ни дорожки.

    Без этой ветки такой материал получал бы нулевую оценку, то есть проходил
    бы гейт при любом объёме.
    """
    settings = _settings(LLM_PROVIDER="anthropic")
    estimate = estimate_from_probe(SourceMeta(text_chars=500_000), settings)
    assert estimate.est_tokens > 0
    assert estimate.est_cost > 0


def test_currency_comes_from_configuration() -> None:
    """Зашитые доллары запрещены §2.1: на этапе 1 валюта долларовая, в
    эксплуатации рублёвая, и разница обнаружится не при смене провайдера, а в
    отчётах за месяц."""
    settings = _settings()
    assert estimate_from_probe(SourceMeta(pages=1), settings).currency == settings.COST_CURRENCY


def test_as_json_keeps_cost_exact() -> None:
    """`est_cost` сериализуется строкой, а не `float`.

    §6.4 сравнивает оценку с фактическим расходом, и потеря точности при
    двойном преобразовании давала бы расхождение там, где его нет.
    """
    estimate = CostEstimate(
        fragments=3, est_tokens=100, est_cost=Decimal("0.1234567"), currency="USD"
    )
    assert estimate.as_json()["est_cost"] == "0.1234567"


def test_refined_estimate_excludes_vision() -> None:
    """Уточнённая оценка не включает vision — он уже оплачен в `extract`.

    §6.4 сравнивает с фактом именно уточнённую оценку. Включив в неё уже
    потраченное, мы сравнивали бы расход со суммой, в которую этот расход
    входит дважды.
    """
    settings = _settings(LLM_PROVIDER="anthropic")
    scanned = estimate_from_probe(SourceMeta(pages=50, vision_pages_ratio=1.0), settings)
    refined = estimate_after_extract(fragments=50, fragment_tokens=25_000, settings=settings)
    assert refined.est_cost < scanned.est_cost


def test_refined_estimate_reports_fragment_count() -> None:
    refined = estimate_after_extract(fragments=42, fragment_tokens=1000, settings=_settings())
    assert refined.fragments == 42


# --- Условия срабатывания -------------------------------------------------


def test_gate_is_disabled_while_threshold_is_unset() -> None:
    """При незаданном `COST_WARN_THRESHOLD` гейт не срабатывает.

    Это решение, а не недосмотр: порог в §30.3 стоит как `TODO(owner)` и
    выводится из спайка (§34.1), который не выполнялся — ключа провайдера нет.
    Срабатывание на `None` остановило бы обработку любого материала, то есть
    продукт не работал бы вовсе до получения ключа.
    """
    settings = _settings()
    assert settings.COST_WARN_THRESHOLD is None
    huge = estimate_from_probe(SourceMeta(pages=10_000, vision_pages_ratio=1.0), settings)
    assert exceeds_threshold(huge, settings) is False


def test_gate_fires_once_threshold_is_set() -> None:
    settings = _settings(COST_WARN_THRESHOLD="1.00", LLM_PROVIDER="anthropic")
    huge = estimate_from_probe(SourceMeta(pages=1000, vision_pages_ratio=1.0), settings)
    small = estimate_from_probe(SourceMeta(pages=1), settings)
    assert exceeds_threshold(huge, settings) is True
    assert exceeds_threshold(small, settings) is False


def test_reestimate_works_even_when_gate_is_disabled() -> None:
    """Повторный вопрос по росту оценки не зависит от абсолютного порога.

    §6.3: «Без этой проверки гейт обходится любым материалом, чью стоимость
    `probe` недооценил». Поэтому вопрос задаётся по относительному росту и
    работает, даже когда `COST_WARN_THRESHOLD` не задан.
    """
    settings = _settings()
    assert settings.COST_WARN_THRESHOLD is None

    preliminary = CostEstimate(0, 1000, Decimal("1.00"), "USD")
    refined = CostEstimate(10, 10_000, Decimal("10.00"), "USD")
    assert needs_reestimate_confirmation(preliminary, refined, settings) is True


def test_reestimate_does_not_fire_on_small_growth() -> None:
    settings = _settings()
    preliminary = CostEstimate(0, 1000, Decimal("1.00"), "USD")
    refined = CostEstimate(10, 1100, Decimal("1.10"), "USD")
    assert needs_reestimate_confirmation(preliminary, refined, settings) is False


def test_reestimate_boundary_uses_configured_factor() -> None:
    """Множитель берётся из `COST_REESTIMATE_FACTOR`, а не зашит.

    Проверяется и ниже, и выше границы: проверка только сверху прошла бы и
    при зашитом множителе любого значения меньше заданного.
    """
    settings = _settings()
    factor = Decimal(str(settings.COST_REESTIMATE_FACTOR))
    base = Decimal("2.00")
    preliminary = CostEstimate(0, 1000, base, "USD")

    just_under = CostEstimate(10, 1000, base * factor, "USD")
    just_over = CostEstimate(10, 1000, base * factor + Decimal("0.01"), "USD")

    assert needs_reestimate_confirmation(preliminary, just_under, settings) is False
    assert needs_reestimate_confirmation(preliminary, just_over, settings) is True


def test_reestimate_handles_zero_preliminary() -> None:
    """Нулевая предварительная оценка не даёт деления на ноль.

    Достижимо: `probe` на источнике, который не удалось измерить, даёт пустой
    `SourceMeta`, то есть нулевую оценку. Без ветки здесь гейт падал бы на
    каждом таком материале.
    """
    settings = _settings()
    zero = CostEstimate(0, 0, Decimal("0"), "USD")
    nonzero = CostEstimate(5, 500, Decimal("0.50"), "USD")
    assert needs_reestimate_confirmation(zero, nonzero, settings) is True
    assert needs_reestimate_confirmation(zero, zero, settings) is False


def test_gate_question_contains_the_numbers() -> None:
    """Цифры в вопросе — не украшение.

    §6.3 задаёт форму сообщения с объёмом, токенами и суммой. Без них выбор
    между «обработать полностью» и «только конспект» делается наугад.
    """
    estimate = CostEstimate(0, 123_456, Decimal("7.89"), "USD")
    text = gate_question(estimate, pages=180)
    assert "180" in text
    assert "123456" in text
    assert "7.89" in text
    assert "USD" in text


def test_coefficients_are_overridable() -> None:
    """Коэффициенты передаются, а не зашиты в расчёт.

    Они предварительные: §6.3 называет «коэффициенты стадий», но нигде их не
    задаёт, и после спайка §34.1 они переедут в §30.3. Возможность подменить
    их — то, что делает этот переезд правкой одного места.
    """
    settings = _settings(LLM_PROVIDER="anthropic")
    cheap = StageCoefficients(text_tokens_per_page=1)
    expensive = StageCoefficients(text_tokens_per_page=10_000)
    meta = SourceMeta(pages=10)

    assert (
        estimate_from_probe(meta, settings, coefficients=cheap).est_cost
        < estimate_from_probe(meta, settings, coefficients=expensive).est_cost
    )


def test_manual_provider_gives_zero_cost_and_nonzero_tokens() -> None:
    """Ручной режим: стоимость ноль, объём не ноль. §2.1, ADR-0005.

    Это не особенность теста, а поведение продукта в том режиме, в котором он
    сейчас работает, и оно должно быть названо проверкой. Ноль в деньгах
    верен: вызова не было. Ноль в токенах был бы неверен: материал через слой
    прошёл, и §6.4 обязан показывать объём работы за весь ручной период.

    Отсюда следствие, записанное в ADR-0005: коридор §6.4 считается на
    токенах. На стоимости он в этом режиме невычислим — `0 / 0`.
    """
    estimate = estimate_from_probe(SourceMeta(pages=10), _settings(LLM_PROVIDER="manual"))

    assert estimate.est_cost == Decimal("0")
    assert estimate.est_tokens > 0


def test_manual_provider_disables_the_cost_gate_even_with_threshold() -> None:
    """При нулевых ценах гейт §6.3 не срабатывает даже при заданном пороге.

    Проверка на вырожденность, а не на желаемое поведение: предупреждать о
    сумме, которой нет, бессмысленно, но знать об этом нужно — оператор,
    выставивший `COST_WARN_THRESHOLD` в ручном режиме, вправе ожидать, что
    порог работает. Вопрос «нужен ли гейт на токенах» вынесен Архитектору
    change request'ом; до решения поведение зафиксировано здесь.
    """
    settings = _settings(COST_WARN_THRESHOLD="0.01", LLM_PROVIDER="manual")
    estimate = estimate_from_probe(SourceMeta(pages=10_000), settings)

    assert estimate.est_tokens > 1_000_000
    assert exceeds_threshold(estimate, settings) is False
