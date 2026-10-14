"""PDF-адаптер. ТЗ §5.1, §5.2, §3.4, §6.3.

**Единственный модуль `core/`, который видит PyMuPDF** — и поэтому
единственный, на который ТЗ §35 (v3.15) даёт послабление
`disallow_untyped_calls = false`. Замер, по которому принято решение: пробный
файл из 18 строк даёт 4 ошибки `--strict` — два вызова `pymupdf.open`, `close`
и итерация `Document`; у библиотеки возврат аннотирован у 150 определений из
1173 при наличии `py.typed`. Держит границу тест
`test_pymupdf_lives_in_one_module`: второй модуль с этим импортом сделает его
красным.

Рендер страницы и рендер с подсветкой — **методы этого же класса**, а не
отдельный модуль. Первая редакция плана выносила их в `core/adapters/render.py`,
и ревью показало, что такой модуль требует второго послабления: `pymupdf.Rect`
и `Pixmap.tobytes` дают три ошибки `--strict`. Послабление «для него одного»
из §35 это бы нарушило.

Раскладка — в `core/adapters/layout.py`, без PyMuPDF: здесь только разбор
документа и сборка локаторов §3.4.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Final

import pymupdf

from core.adapters.base import ExtractedFragment, IngestCtx, SourceAdapter, SourceMeta
from core.adapters.layout import Block, HeadingPlan, analyze_headings, glue, is_scanned
from core.config import Settings
from core.llm.estimation import tokens_in
from core.logging import get_logger

log = get_logger(__name__)

RENDER_DPI: Final = 150
"""§5.2 п. 2 и §23: страница рендерится в PNG 150 dpi.

Одно число на оба случая — путь через vision и подсветку для перехода к
источнику: разные значения дали бы человеку картинку одного разрешения, а
модели другого, и расхождение всплыло бы на сравнении координат.
"""

HIGHLIGHT_COLOR: Final = (1.0, 0.0, 0.0)
HIGHLIGHT_WIDTH: Final = 1.5

VISION_PENDING_KEY: Final = "vision_pending_pages"
"""Ключ `materials.origin` со списком страниц, ждущих vision.

Нужен, потому что до WP-06 PR B сканированные страницы пропускаются, а §6.3
требует, чтобы последствия были видимыми: материал, из которого молча выпали
страницы, доходит до `ready`, и человек не может понять, что у него на руках.
Ключ печатает `make dump-material`, и его число называет сообщение о готовности.
"""


class PdfAdapter(SourceAdapter):
    """Текстовый слой PDF, локаторы §3.4 и рендер страниц.

    Страницы-сканы в этом пакете **пропускаются** и перечисляются в
    `SourceMeta.extra[VISION_PENDING_KEY]`; обработка приедет в PR B вместе с
    vision. Если сканы — все страницы, фрагментов не будет вовсе, и стадия
    упадёт на проверке `core/ingest/handlers.py` с внятной причиной: материал
    уйдёт в `failed`, а не в `ready` без содержания.
    """

    kind = "pdf"

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    async def probe(self, path: str) -> SourceMeta:
        """Число страниц, объём текста и доля страниц под vision. §5.1, §6.3.

        Дешёвый по требованию §5.1: читает только текстовый слой, модель не
        вызывает. Доля страниц под vision — главный множитель оценки §6.3
        (`core/ingest/budget.py`), и считать её нужно **до** `extract`.
        """
        document = pymupdf.open(path)
        try:
            pages = document.page_count
            chars = 0
            pending: list[int] = []
            for index in range(pages):
                page = document.load_page(index)
                text = page.get_text("text")
                chars += len(text)
                if is_scanned(len(text), has_images=bool(page.get_images())):
                    pending.append(index + 1)
        finally:
            document.close()

        ratio = len(pending) / pages if pages else 0.0
        log.info(
            "pdf_probed",
            pages=pages,
            text_chars=chars,
            vision_pages=len(pending),
        )
        return SourceMeta(
            pages=pages,
            text_chars=chars,
            vision_pages_ratio=ratio,
            extra={"pages": pages, VISION_PENDING_KEY: pending},
        )

    async def _fragments(self, path: str) -> list[ExtractedFragment]:
        """Текстовые фрагменты всего документа. Вынесено ради `extract`."""
        document = pymupdf.open(path)
        try:
            blocks: list[Block] = []
            for index in range(document.page_count):
                page = document.load_page(index)
                text = page.get_text("text")
                if is_scanned(len(text), has_images=bool(page.get_images())):
                    # Пропуск назван в `probe` ключом `vision_pending_pages`;
                    # обработка — PR B.
                    log.info("pdf_page_awaits_vision", page=index + 1)
                    continue
                blocks.extend(_blocks_of(page, index + 1))
        finally:
            document.close()

        plan: HeadingPlan = analyze_headings(blocks)
        pieces = glue(blocks, count_tokens=tokens_in, plan=plan)

        out: list[ExtractedFragment] = []
        for ord_, piece in enumerate(pieces, start=1):
            locator: dict[str, Any] = {
                "type": "pdf",
                # Человеческий номер, с единицы: он попадает человеку в разбор
                # §22 и по нему же рендерится подсветка §23. PyMuPDF нумерует
                # с нуля, и конвертация сделана в `_blocks_of` — единственном
                # месте. Основание — ADR-0007.
                "page": piece.page,
                "bbox": list(piece.bbox),
            }
            if piece.heading is not None:
                locator["heading"] = piece.heading
            out.append(
                ExtractedFragment(
                    ord=ord_,
                    kind="text",
                    locator=locator,
                    text=piece.text,
                    quality=1.0,
                    tokens=piece.tokens,
                )
            )
        return out

    async def extract(self, path: str, ctx: IngestCtx) -> AsyncIterator[ExtractedFragment]:
        """Фрагменты документа. §5.1: асинхронный итератор.

        `ctx` пока не используется: временная директория нужна изображениям, а
        они приезжают в PR C. Параметр оставлен, потому что он в контракте
        §5.1, и менять контракт ради одного пакета нельзя.
        """
        for fragment in await self._fragments(path):
            yield fragment

    def render_page(self, path: str, page: int) -> bytes:
        """Страница в PNG 150 dpi. §5.2 п. 2, §23.

        `page` — человеческий номер, с единицы, как в локаторе §3.4. Номер вне
        документа — `ValueError`; почему отказ, а не чужая страница, сказано у
        `_page_index`.
        """
        document = pymupdf.open(path)
        try:
            index = _page_index(page, document.page_count)
            pixmap = document.load_page(index).get_pixmap(dpi=RENDER_DPI)
            return bytes(pixmap.tobytes("png"))
        finally:
            document.close()

    def render_highlighted(self, path: str, page: int, bbox: Sequence[float]) -> bytes:
        """Страница с подсвеченной областью. §23, критерий приёмки WP-06.

        Рамка рисуется поверх страницы перед рендером, сам файл не меняется:
        документ открыт на чтение и закрывается без сохранения.
        """
        document = pymupdf.open(path)
        try:
            target = document.load_page(_page_index(page, document.page_count))
            target.draw_rect(pymupdf.Rect(*bbox), color=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH)
            pixmap = target.get_pixmap(dpi=RENDER_DPI)
            return bytes(pixmap.tobytes("png"))
        finally:
            document.close()


def _page_index(page: int, page_count: int) -> int:
    """Человеческий номер страницы → индекс PyMuPDF. Вне документа — отказ.

    Проверка нужна не от небрежности вызывающего. PyMuPDF адресует
    отрицательные индексы **с конца документа**, поэтому без неё `page = 0`
    даёт `load_page(-1)` — последнюю страницу, без исключения. Замер на
    двухстраничной фикстуре: `page = 0` отдавал байт-в-байт страницу 2, а
    `page = -1` — страницу 1. Ни один ассерт такую подмену от правильного
    ответа не отличит, а §23 и WP-11 будут рендерить по номерам **из базы**, а
    не по только что собранным. Основание — ADR-0007 решение 3.
    """
    if not 1 <= page <= page_count:
        raise ValueError(
            f"номер страницы {page} вне документа: страницы нумеруются "
            f"с единицы, их всего {page_count}"
        )
    return page - 1


def _blocks_of(page: Any, human_page: int) -> list[Block]:
    """Текстовые блоки страницы в структуры раскладки.

    Здесь и только здесь номер страницы переводится из нумерации PyMuPDF (с
    нуля) в человеческую (с единицы): дальше о нуле не знает никто.
    """
    out: list[Block] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = block.get("lines") or []
        text = " ".join(
            span["text"] for line in lines for span in (line.get("spans") or [])
        ).strip()
        if not text:
            continue
        sizes = [float(span["size"]) for line in lines for span in (line.get("spans") or [])]
        out.append(
            Block(
                page=human_page,
                text=text,
                bbox=tuple(float(value) for value in block["bbox"]),  # type: ignore[arg-type]
                size=max(sizes) if sizes else 0.0,
            )
        )
    return out


def is_pdf(path: str) -> bool:
    """Расширение — `.pdf`. Используется фабрикой адаптеров."""
    return Path(path).suffix.lower() == ".pdf"
