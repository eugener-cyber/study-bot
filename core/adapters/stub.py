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
from dataclasses import replace
from pathlib import Path
from typing import Final

from core.adapters.base import (
    ExtractedAsset,
    ExtractedFragment,
    IngestCtx,
    SourceAdapter,
    SourceMeta,
)

PNG_STUB: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
"""Однопиксельный PNG: содержимое файла ассета у заглушки.

Настоящие байты, а не произвольные: тест может открыть файл и убедиться, что
перенос §4.2 сохранил его целым, а не только создал имя."""


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
        assets: list[ExtractedAsset] | None = None,
    ) -> None:
        self._meta = meta if meta is not None else SourceMeta(pages=1, text_chars=100)
        self._fragments = fragments if fragments is not None else [_default_fragment()]
        self._assets = assets if assets is not None else []
        """Ассеты, которые отдаст `assets()`. По умолчанию ни одного.

        Задаются явно, как и фрагменты: стадии нужно уметь показать «два
        изображения, одно в наборе», а повторному запуску — «теперь одно».
        Пустой список по умолчанию оставляет поведение заглушки прежним для
        текстовых материалов.
        """
        self.probe_calls = 0
        self.extract_calls = 0
        """Счётчики вызовов.

        Нужны тесту гейта бюджета: проверять надо не «гейт сработал», а
        «`extract` не вызывался». Первая формулировка прошла бы и при гейте,
        поставленном после `extract`, — то есть при том самом дефекте, из-за
        которого §6.3 переписывался в v3.5.
        """

    async def assets(self, path: str, ctx: IngestCtx) -> AsyncIterator[ExtractedAsset]:
        """Пишет заданные ассеты в `ctx.tmp_dir` и отдаёт их с новыми путями.

        Файл создаётся по-настоящему: стадия кладёт в базу путь и полагается
        на то, что конвейер перенесёт файл (§4.2). Заглушка, которая только
        отдаёт структуры, позволила бы тесту «файл существует по пути из базы»
        пройти на пустом каталоге.
        """
        for asset in self._assets:
            target = Path(ctx.tmp_dir) / Path(asset.path).name
            target.write_bytes(PNG_STUB)
            yield replace(asset, path=str(target))

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
