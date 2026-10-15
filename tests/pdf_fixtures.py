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


CYRILLIC_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
"""Где искать шрифт с кириллицей для подписей «Рис. N».

Встроенные шрифты PyMuPDF (base-14) кириллицы не несут — замер: «Рис. 1»
извлекается как «···. 1». Файл шрифта в репозиторий не кладётся (ADR-0007
решение 7), поэтому берётся системный, а тест, которому он нужен,
пропускается, если ни одного нет. Проверяется именно то, ради чего кириллица
нужна: что подпись доходит из разбора PDF до правила набора.
"""


def cyrillic_font() -> str | None:
    """Путь к системному шрифту с кириллицей или `None`."""
    for candidate in CYRILLIC_FONTS:
        if Path(candidate).is_file():
            return candidate
    return None


def _fill_text_layer(page: pymupdf.Page, text: str) -> None:
    """Добавляет странице текстовый слой длиннее `SCAN_TEXT_THRESHOLD`.

    Без него страница с картинками подпадает под детект скана §5.2 п. 2 —
    «менее 100 символов текста при непустом изображении», — и адаптер её
    пропустит. На фикстурах PR A этот же урок стоил зелёного прогона: текста в
    них заведомо больше порога, и здесь по той же причине.
    """
    page.insert_textbox(
        pymupdf.Rect(60, page.rect.height - 180, page.rect.width - 60, page.rect.height - 60),
        text,
        fontsize=BODY_SIZE,
        fontname="helv",
    )


def _png_bytes(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    """Однотонный PNG нужного размера. Через PyMuPDF, чтобы не плодить зависимости."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pixmap.set_rect(pixmap.irect, colour)
    return bytes(pixmap.tobytes("png"))


def figures_pdf(path: Path, *, caption_prefix: str = "Ris.") -> Path:
    """Страница с набором «Рис. 1, 2, 3» и страница с одиночным изображением.

    Собирается кодом, как и остальные фикстуры (ADR-0007 решение 7). Что в ней
    важно для §5.2 п. 4:

    - три мелких изображения в ряд, под каждым своя подпись «Ris. N» — то есть
      **первый** признак набора, и площадь каждого ниже порога §30.3;
    - три картинки набора **разные**: одинаковые байты PDF хранит одним
      объектом, и дедупликация по `xref` схлопнула бы набор в один ассет — на
      этом первая редакция фикстуры и попалась;
    - на второй странице одно крупное изображение без подписи, размещённое
      **дважды** и оба раза выше порога: так дедупликация проверяется
      независимо от подписей (замер корпуса: 86 лишних размещений, максимум
      31 копия).
    """
    document = pymupdf.open()
    first = document.new_page()
    font = cyrillic_font()
    kwargs = (
        {"fontname": "cap", "fontfile": font}
        if caption_prefix != "Ris." and font is not None
        else {"fontname": "helv"}
    )
    # Три **разных** изображения: одинаковые байты PDF хранит одним объектом, и
    # дедупликация по `xref` схлопнула бы набор в один ассет — на этом первая
    # редакция фикстуры и попалась.
    for index, x in enumerate((60.0, 200.0, 340.0), start=1):
        element = _png_bytes(40, 40, (10 * index, 60, 200))
        first.insert_image(pymupdf.Rect(x, 100, x + 40, 140), stream=element)
        first.insert_text((x, 160), f"{caption_prefix} {index}. Element", fontsize=9, **kwargs)
    # Повторное размещение **первого** изображения — предмет дедупликации.
    first.insert_image(pymupdf.Rect(60, 300, 100, 340), stream=_png_bytes(40, 40, (10, 60, 200)))
    _fill_text_layer(first, BODY_ONE)

    second = document.new_page()
    big = _png_bytes(300, 300, (220, 40, 40))
    # Дважды одно и то же изображение, и оба размещения **выше** порога §30.3:
    # дедупликация видна без подписей, то есть и на фикстуре с латиницей.
    second.insert_image(pymupdf.Rect(100, 100, 400, 400), stream=big)
    second.insert_image(pymupdf.Rect(100, 450, 400, 750), stream=big)
    _fill_text_layer(second, BODY_TWO)

    document.save(str(path))
    document.close()
    return path


def oversized_image_pdf(path: Path, *, side: int = 1800) -> Path:
    """Страница с изображением, длинная сторона которого больше 1568 px.

    1800, а не 3000: однотонный PNG 3000×1500 даёт файл на 13 МБ и замедляет
    прогон, а для проверки ресайза достаточно превысить предел.
    """
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=_png_bytes(side, side // 2, (30, 160, 90)))
    _fill_text_layer(page, BODY_ONE)
    document.save(str(path))
    document.close()
    return path


def repeated_across_pages_pdf(path: Path, *, pages: int = 3) -> Path:
    """Одно изображение на нескольких страницах по одному размещению.

    Главный случай дедупликации, и он **не** внутристраничный: замер по
    `tasksans-tech-9-teor-kddt-mun-mosobl-25-26.pdf` — белая подложка лежит на
    31 разной странице по одному размещению, и область «внутри страницы»
    давала 41 ассет вместо 11. Первая редакция фикстуры повторяла картинку на
    одной странице и этот случай не проверяла (нашло ревью PR #48).

    Картинка крупная: порог §30.3 её пропускает, то есть проверяется именно
    дедупликация, а не отбор по площади.
    """
    document = pymupdf.open()
    big = _png_bytes(300, 300, (40, 120, 220))
    for index in range(pages):
        page = document.new_page()
        page.insert_image(pymupdf.Rect(100, 100, 400, 400), stream=big)
        _fill_text_layer(page, f"{BODY_ONE} Page {index + 1}.")
    document.save(str(path))
    document.close()
    return path
