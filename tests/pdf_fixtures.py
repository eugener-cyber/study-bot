"""Фикстуры PDF, которые строятся кодом. ТЗ §34 (v3.14), решение по WP-06.

Бинарников в репозитории нет намеренно: двухстраничный файл с картинками
собирается десятком строк, а 600 КБ материала заказчика в публичном репозитории
— это и лишний вес, и его данные.

Синтетический скан — решение Архитектора 2026-10-14 (Issue #38): сканов в
материалах нет, проверены все 49 файлов, в самом разреженном 556 символов
текста на страницу. Страница рендерится в PNG и вставляется в PDF **без**
текстового слоя, то есть детект §5.2 п. 2 получает ровно то, что получил бы на
настоящем скане: ноль символов при непустом изображении.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_ONE_MARKER = "PAGE-ONE-MARKER"
PAGE_TWO_MARKER = "PAGE-TWO-MARKER"
"""Маркеры в тексте страниц.

По ним номер страницы в локаторе закрепляется **против содержания**, а не
против обратного преобразования: пара «прибавить единицу при разборе, вычесть
при рендере» самосогласована, и тест через неё зелен при симметричной ошибке.
"""

HEADING_SIZE = 18.0
BODY_SIZE = 10.0

BODY_ONE = (
    "Chalk marks the fabric on the face side and brushes off without a trace. "
    "The mark must stay visible until the cut is finished, otherwise the line "
    "has to be drawn again on a shifted layer."
)
BODY_TWO = (
    "The seam is secured at the start and at the end of the stitch line. "
    "Without the lock stitch the thread unravels from the very first wash, "
    "and the seam opens along the whole length."
)
"""Тексты страниц — латиницей, и это не небрежность.

Встроенные шрифты PyMuPDF (`helv` и прочие base-14) кириллицы не несут:
вставленный русский текст извлекается как последовательность точек, то есть
маркер страницы перестаёт быть маркером. Настоящий кириллический шрифт
означал бы файл шрифта в репозитории — ровно тот бинарник, которого фикстуры,
строящиеся кодом, и позволяют избежать.

Длина обоих текстов заведомо больше `SCAN_TEXT_THRESHOLD` (100 символов,
§5.2 п. 2): иначе текстовая страница сама подошла бы под детект скана, и
проверка путей «текст» и «скан» перестала бы их различать.
"""


def text_pdf(path: Path) -> Path:
    """Две страницы: заголовок кеглем 18 и абзацы кеглем 10 под ним."""
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 90), "Fabric cutting", fontsize=HEADING_SIZE, fontname="helv")
    first.insert_textbox(
        pymupdf.Rect(72, 120, 523, 260),
        f"{PAGE_ONE_MARKER}. {BODY_ONE}",
        fontsize=BODY_SIZE,
        fontname="helv",
    )
    second = document.new_page()
    second.insert_text((72, 90), "Stitching", fontsize=HEADING_SIZE, fontname="helv")
    second.insert_textbox(
        pymupdf.Rect(72, 120, 523, 260),
        f"{PAGE_TWO_MARKER}. {BODY_TWO}",
        fontsize=BODY_SIZE,
        fontname="helv",
    )
    document.save(str(path))
    document.close()
    return path


def scanned_pdf(path: Path, *, source: Path | None = None) -> Path:
    """Синтетический скан: страница картинкой, текстового слоя нет.

    Собирается из страницы текстового PDF, то есть содержит настоящий текст —
    но только как изображение. Для детекта §5.2 п. 2 это и есть скан: ноль
    символов при непустом изображении.
    """
    origin = source or text_pdf(path.with_name("source-for-scan.pdf"))
    document = pymupdf.open(str(origin))
    try:
        pixmap = document.load_page(0).get_pixmap(dpi=150)
    finally:
        document.close()

    scan = pymupdf.open()
    page = scan.new_page(width=pixmap.width, height=pixmap.height)
    page.insert_image(page.rect, pixmap=pixmap)
    scan.save(str(path))
    scan.close()
    return path


def mixed_pdf(path: Path) -> Path:
    """Первая страница текстовая, вторая — скан.

    На таком файле проверяется, что §6.3 не нарушается: материал доходит до
    `ready`, но пропущенная страница названа в `materials.origin`, а не
    потеряна молча.
    """
    text_source = text_pdf(path.with_name("source-for-mixed.pdf"))
    document = pymupdf.open(str(text_source))
    try:
        pixmap = document.load_page(1).get_pixmap(dpi=150)
    finally:
        document.close()

    mixed = pymupdf.open(str(text_source))
    try:
        mixed.delete_page(1)
        page = mixed.new_page(width=pixmap.width, height=pixmap.height)
        page.insert_image(page.rect, pixmap=pixmap)
        mixed.save(str(path))
    finally:
        mixed.close()
    return path


_MINIMAL_PDF = """%PDF-1.7
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[]/Count {count}>>endobj
trailer<</Root 1 0 R>>
%%EOF
"""
"""Каркас PDF без страниц, в котором `/Count` задаётся снаружи.

Две поломки из одного шаблона, и обе — не «мусор вместо файла», а правильно
устроенный документ с неверным числом страниц: такой открывается, а падает
позже. Этим он и ценен — именно на нём `pymupdf.open` проходит, а
`document.page_count` поднимает голый `RuntimeError`.
"""


def lying_page_count_pdf(path: Path) -> Path:
    """PDF, обещающий три страницы при пустом `Kids`.

    Замер: `pymupdf.open` проходит, `document.page_count` поднимает
    `RuntimeError: code=7: Invalid number of pages` — **без** своего класса у
    библиотеки. Тот же отказ давал обрезанный настоящий PDF; обрезание
    синтетической фикстуры его не даёт (PyMuPDF восстанавливает xref
    маленького файла), поэтому поломка собрана прямо, а не отрезанием.
    """
    path.write_text(_MINIMAL_PDF.format(count=3), encoding="ascii")
    return path


def zero_page_pdf(path: Path) -> Path:
    """Корректный PDF без страниц: открывается, `page_count` равен нулю.

    Извлекать из него нечего, и `probe` без проверки отдал бы `pages = 0`, а
    материал дошёл бы до `ready` без содержания.
    """
    path.write_text(_MINIMAL_PDF.format(count=0), encoding="ascii")
    return path
