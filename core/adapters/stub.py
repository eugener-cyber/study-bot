"""Адаптер-заглушка — для сквозного прогона конвейера без настоящего источника.

Нужен, потому что критерий приёмки WP-03 звучит так: «`make seed` доводит
материал до `ready` на заглушках». Машина стадий, атомарность §4.2,
возобновление §4.3 и гейт бюджета §6.3 проверяются здесь целиком, и ни одна
из этих проверок не должна зависеть от того, умеет ли проект читать PDF.
Настоящий PDF-адаптер — WP-06.

Это **не** костыль по п. 1 списка (заглушка вместо реализации): заглушка
объявлена планом работ как часть пакета, живёт в отдельном модуле и не
подменяет собой настоящий адаптер — она регистрируется под своим `kind`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from core.adapters.base import ExtractedFragment, IngestCtx, SourceAdapter, SourceMeta


class StubAdapter(SourceAdapter):
    """Отдаёт то, что ему задали при создании.

    Значения задаются явно, а не генерируются: тест гейта бюджета должен
    уметь задать «двести страниц, половина сканы», а тест пересчёта оценки —
    расхождение между `probe` и фактическим числом фрагментов. Случайные или
    выведенные значения такие сценарии выразить не дают.
    """

    kind = "text"

    def __init__(
        self,
        *,
        meta: SourceMeta | None = None,
        fragments: list[ExtractedFragment] | None = None,
    ) -> None:
        self._meta = meta if meta is not None else SourceMeta(pages=1, text_chars=100)
        self._fragments = fragments if fragments is not None else [_default_fragment()]
        self.probe_calls = 0
        self.extract_calls = 0
        """Счётчики вызовов.

        Нужны тесту гейта бюджета: проверять надо не «гейт сработал», а
        «`extract` не вызывался». Первая формулировка прошла бы и при гейте,
        поставленном после `extract`, — то есть при том самом дефекте, из-за
        которого §6.3 переписывался в v3.5.
        """

    async def probe(self, path: str) -> SourceMeta:
        self.probe_calls += 1
        return self._meta

    def extract(self, path: str, ctx: IngestCtx) -> AsyncIterator[ExtractedFragment]:
        self.extract_calls += 1
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ExtractedFragment]:
        for fragment in self._fragments:
            yield fragment


def _default_fragment() -> ExtractedFragment:
    return ExtractedFragment(
        ord=1,
        kind="text",
        locator={"type": "web", "url": "stub://material", "quote": "заглушка"},
        text="Текст фрагмента-заглушки.",
        quality=1.0,
        tokens=10,
    )
