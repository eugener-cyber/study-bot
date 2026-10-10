"""Объёмы и цены вызовов. ТЗ §2.1, §6.3, §6.4.

Числа здесь записаны **литералами**, а не взяты из рабочего кода. Урок WP-01:
фикстура, собранная из тех же констант, что проверяет, проходит при сломанном
коде — она сравнивает значение с собой. Поэтому смена цены или версии тарифа
требует правки этого файла, и правка видна в диффе.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.llm.estimation import (
    CHARS_PER_PAGE,
    PRICES,
    PRICING_VERSION,
    StageCoefficients,
    prices_for,
    tokens_in,
    tokens_per_page,
)


def test_pricing_version_and_numbers_are_pinned_together() -> None:
    """Версия тарифа и сами цены меняются вместе (§3, §6.4).

    Правка цен без правки версии пометила бы всю прошлую историю `llm_calls`
    неверно — то есть сделала бы бесполезным именно то поле, ради которого §3
    ввёл `pricing_version`.
    """
    assert PRICING_VERSION == "2026-10-anthropic-opus5"
    assert PRICES["anthropic"].input_per_million == Decimal("5")
    assert PRICES["anthropic"].output_per_million == Decimal("25")


def test_manual_prices_are_zero_and_yandex_is_absent() -> None:
    """§2.1 после решения Архитектора: у `manual` стоимость ноль.

    Запись `yandexgpt` отсутствует намеренно: §2.1 говорит «по тарифам
    Яндекса», чисел не называет, и ноль здесь выглядел бы бесплатным
    провайдером — то есть §6.4 показывал бы нулевой расход при реальных
    счетах.
    """
    assert PRICES["manual"].cost(1_000_000, 1_000_000) == Decimal("0")
    assert "yandexgpt" not in PRICES

    with pytest.raises(ValueError, match="yandexgpt"):
        prices_for("yandexgpt")


def test_cost_matches_the_prices_of_section_2_1() -> None:
    """Миллион входа и миллион выхода — $5 и $25 (§2.1)."""
    assert prices_for("anthropic").cost(1_000_000, 0) == Decimal("5")
    assert prices_for("anthropic").cost(0, 1_000_000) == Decimal("25")
    assert prices_for("anthropic").cost(1_000, 500) == Decimal("0.0175")


def test_tokens_per_page_is_derived_not_duplicated() -> None:
    """Токенов на странице выводится из коэффициента, а не задаётся вторым числом.

    Два независимых числа для одного и того же разошлись бы, и §6.3 оценивал
    бы материал по одному, а §6.4 те же вызовы по другому — коридор приёмки
    показал бы расхождение там, где его нет. Это уже случалось: редакция 2
    плана WP-04 предлагала 2.5 символа на токен при 5.0 в WP-03.
    """
    assert tokens_per_page() == 500
    assert StageCoefficients().text_tokens_per_page == tokens_per_page()
    assert CHARS_PER_PAGE == 2500


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("я" * 100, 20),
        ("a" * 100, 25),
    ],
)
def test_tokens_in_uses_the_alphabet_of_the_text(text: str, expected: int) -> None:
    """Коэффициент выбирается по доле кириллицы, а не по языку материала."""
    assert tokens_in(text) == expected


def test_tokens_in_blends_mixed_alphabets() -> None:
    """Смешанный текст получает промежуточную оценку, а не одну из крайних.

    Проверка именно на «между»: оба коэффициента по отдельности дали бы
    систематический перекос в сторону того алфавита, которого в примере
    оказалось больше, а в учебном тексте перемешаны термины и формулы.
    """
    mixed = "я" * 50 + "a" * 50
    assert tokens_in("a" * 100) > tokens_in(mixed) > tokens_in("я" * 100)


def test_short_text_never_costs_zero_tokens() -> None:
    """Непустой текст стоит хотя бы один токен.

    Иначе оценка вызова по короткому промпту давала бы ноль, а §6.4 делит на
    оценку: `estimated_cost = 0` делает коридор невычислимым — ровно тот
    дефект, на котором упёрлась редакция 2 плана.
    """
    assert tokens_in("а") == 1
    assert tokens_in(".") == 1


def test_output_ratio_is_known_for_every_stage_that_calls_the_model() -> None:
    """Три стадии, обращающиеся к модели, имеют долю выхода."""
    coefficients = StageCoefficients()
    for purpose in ("sections", "facts", "questions"):
        assert coefficients.output_ratio_for(purpose) > 0


def test_notes_has_no_output_ratio() -> None:
    """Конспект не вызывает модель, значит доли выхода у него быть не должно.

    Решение Архитектора v3.13 (CR-O, #26). Коэффициент, оставленный «на
    всякий случай», держал бы оценку расхода для вызова, которого не бывает:
    первые реальные материалы показали бы систематическое завышение, а
    причину искали бы в ценах. Исключение вместо умолчания здесь и работает
    как проверка: вызов с назначением `notes` — дефект, и он виден сразу.
    """
    with pytest.raises(ValueError, match="notes"):
        StageCoefficients().output_ratio_for("notes")


def test_unknown_purpose_raises_instead_of_defaulting() -> None:
    """Неизвестное назначение — исключение, а не умолчание.

    Умолчание дало бы оценку, похожую на правильную, и расхождение в §6.4
    искали бы в ценах — то есть в последнем месте, где оно есть.
    """
    with pytest.raises(ValueError, match="неизвестное назначение"):
        StageCoefficients().output_ratio_for("summary")


def test_unknown_provider_raises_instead_of_free_calls() -> None:
    with pytest.raises(ValueError, match="нет тарифов"):
        prices_for("openai")
