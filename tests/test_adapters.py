"""Контракт источника §5.1 и адаптер-заглушка.

Без базы и без сети: здесь проверяется форма интерфейса, а не работа с
материалом. Настоящий PDF-адаптер — WP-06, и эти же тесты станут для него
контрактом.
"""

from __future__ import annotations

import inspect

import pytest

from core.adapters.base import ExtractedFragment, IngestCtx, SourceAdapter, SourceMeta
from core.adapters.stub import StubAdapter


def test_adapter_cannot_be_instantiated_without_both_methods() -> None:
    """§5.1 объявляет ровно два метода, и оба обязательны.

    Адаптер без `probe` сделал бы невозможным гейт бюджета §6.3, а без
    `extract` — саму обработку. Проверяется через `ABC`, а не чтением
    сигнатур: иначе «реализация» без метода прошла бы.
    """

    class Incomplete(SourceAdapter):
        kind = "text"

        async def probe(self, path: str) -> SourceMeta:
            return SourceMeta()

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_extract_is_not_a_coroutine_function() -> None:
    """`extract` возвращает итератор, а не корутину.

    §5.1 пишет `def extract(...) -> AsyncIterator[Fragment]`, без `async`.
    Разница существенная: материал может дать тысячи фрагментов, и
    `async def`, возвращающий список, держал бы их все в памяти.
    """
    assert not inspect.iscoroutinefunction(SourceAdapter.extract)
    assert inspect.iscoroutinefunction(SourceAdapter.probe)


def test_probe_meta_allows_missing_fields() -> None:
    """Пустой `SourceMeta` допустим.

    У PDF есть страницы и нет длительности, у аудио наоборот. Пустой означает
    «измерить не удалось» — тогда оценка §6.3 строится по размеру файла, а не
    выдумывается.
    """
    meta = SourceMeta()
    assert meta.pages is None
    assert meta.duration_sec is None
    assert meta.vision_pages_ratio == 0.0


def test_vision_ratio_defaults_to_zero() -> None:
    """Доля страниц под vision по умолчанию ноль.

    Умолчание выбрано в сторону дешёвой оценки намеренно: завышенная оценка
    упёрлась бы в гейт §6.3 и остановила обработку материала, который ничего
    не стоит. Заниженная приведёт к повторному вопросу по
    `COST_REESTIMATE_FACTOR` — то есть к тому же гейту, но позже и по факту.
    """
    assert SourceMeta(pages=10).vision_pages_ratio == 0.0


def test_fragment_requires_locator() -> None:
    """Локатор обязателен: без него нет перехода к источнику (§3, §3.4)."""
    with pytest.raises(TypeError):
        ExtractedFragment(ord=1, kind="text")  # type: ignore[call-arg]


def test_fragment_quality_defaults_to_one() -> None:
    """`quality` по умолчанию 1.0 — как `DEFAULT 1.0` в §3.

    Это множитель `final_confidence` (§7.3): значение по умолчанию не должно
    занижать уверенность факта, иначе адаптер, не умеющий оценивать качество,
    молча исключал бы факты из обучаемых.
    """
    fragment = ExtractedFragment(ord=1, kind="text", locator={"type": "web"})
    assert fragment.quality == 1.0


async def test_stub_returns_what_it_was_given() -> None:
    meta = SourceMeta(pages=200, vision_pages_ratio=0.5)
    fragments = [
        ExtractedFragment(ord=1, kind="text", locator={"type": "web"}, text="раз"),
        ExtractedFragment(ord=2, kind="text", locator={"type": "web"}, text="два"),
    ]
    adapter = StubAdapter(meta=meta, fragments=fragments)

    assert await adapter.probe("x") is meta
    collected = [f async for f in adapter.extract("x", IngestCtx(1, "/tmp"))]
    assert [f.text for f in collected] == ["раз", "два"]


async def test_stub_counts_calls() -> None:
    """Счётчики нужны тесту гейта бюджета.

    Проверять надо не «гейт сработал», а «`extract` не вызывался»: первая
    формулировка прошла бы и при гейте после `extract` — то есть при том
    дефекте, из-за которого §6.3 переписывался в v3.5.
    """
    adapter = StubAdapter()
    assert (adapter.probe_calls, adapter.extract_calls) == (0, 0)

    await adapter.probe("x")
    assert (adapter.probe_calls, adapter.extract_calls) == (1, 0)

    async for _ in adapter.extract("x", IngestCtx(1, "/tmp")):
        pass
    assert (adapter.probe_calls, adapter.extract_calls) == (1, 1)


def test_stub_declares_its_own_kind() -> None:
    """Заглушка не подменяет настоящий адаптер — у неё свой `kind`.

    Иначе она была бы заглушкой вместо реализации (п. 1 списка костылей), а
    не отдельным источником, объявленным планом работ.
    """
    assert StubAdapter.kind == "text"
