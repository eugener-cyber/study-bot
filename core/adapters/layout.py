"""Раскладка страницы: блоки → фрагменты, заголовки, группы изображений.

ТЗ §5.2 пункты 2–5. Здесь **нет** PyMuPDF и нет обращений к базе: на вход
приходят простые структуры, на выход — заготовки фрагментов. Причина не
стилистическая. Границу 600–900 токенов, правило группы «Рис. N» и эвристику
заголовков нужно проверять на десятке искусственных раскладок, и каждая такая
проверка через сконструированный PDF стоила бы файла на диске и зависела бы от
версии библиотеки. Разбор самого PDF живёт в `core/adapters/pdf.py` — он же
единственный модуль `core/`, который видит PyMuPDF (ТЗ §35, послабление
`mypy`).

Номера страниц здесь уже **человеческие, с единицы**: конвертация из нумерации
PyMuPDF сделана в адаптере, и ниже о нуле не знает никто. Основание — ADR-0007.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from core.logging import get_logger

log = get_logger(__name__)

MAX_FRAGMENT_TOKENS: Final = 900
"""§5.2 п. 3 (v3.16): верхняя граница **склейки**. Обязательна для неё.

Границу ограничивает склейка, а не абзац: один блок длиннее 900 токенов
разрезать «по границам абзацев» невозможно, и он уходит во фрагмент целиком с
записью в журнал (`block_exceeds_max_tokens`). Решение Архитектора 2026-10-14
по замечанию ревью PR #42: резать такой блок по предложениям или по счёту
токенов запрещено — `bbox` фрагмента есть объединение bbox его блоков, у двух
частей одного блока он был бы одинаковым, и подсветка §23 указывала бы на весь
абзац вместо нужного места.

Замер на проверенном материале: 96 блоков, максимум 153 токена, медиана 17.5 —
то есть ветвь редкая, но её последствие теперь названо, а не выбрано молча.
"""

MIN_FRAGMENT_TOKENS: Final = 600
"""§5.2 п. 3: нижняя граница — **не** обязательна.

«На разреженном материале это означает фрагменты меньше 600 токенов — верхняя
граница обязательна, нижняя нет». Поэтому значение участвует только в решении
«начинать ли новый фрагмент», а не в проверке результата.
"""

SCAN_TEXT_THRESHOLD: Final = 100
"""§5.2 п. 2: «менее 100 символов текста при непустом изображении» — скан."""

FIGURE_CAPTION = re.compile(r"^\s*рис\.?\s*(\d+)", re.IGNORECASE)
"""Подпись пронумерованного набора §5.2 п. 4.

Форма «Рис. N» взята из спецификации дословно; точка необязательна, регистр
не важен. Другие формы подписи (`Фото 1`, `Схема 2`) сюда не попадают
намеренно: §5.2 называет одну, а догадываться за спецификацию — §40.
"""


@dataclass(frozen=True)
class Block:
    """Текстовый блок страницы: то, что отдаёт `page.get_text("dict")`."""

    page: int
    """Человеческий номер страницы, с единицы."""
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    """Максимальный кегль в блоке — по нему работает эвристика §5.2 п. 5."""


@dataclass(frozen=True)
class Piece:
    """Заготовка текстового фрагмента: всё, кроме идентификаторов."""

    page: int
    text: str
    bbox: tuple[float, float, float, float]
    tokens: int
    heading: str | None


@dataclass(frozen=True)
class Picture:
    """Изображение страницы до решения о сохранении."""

    page: int
    bbox: tuple[float, float, float, float]
    area_ratio: float
    """Доля площади страницы. Сравнивается с `MIN_IMAGE_AREA_RATIO` (§30.3)."""
    caption: str | None = None
    """Текст рядом с изображением, если он найден: по нему работает правило
    группы §5.2 п. 4."""


@dataclass(frozen=True)
class KeptPicture:
    """Изображение, которое сохраняется, и его группа."""

    picture: Picture
    group_id: str | None
    """Непустой у изображений пронумерованного набора «Рис. N» (§3.4, v3.15)."""


def union_bbox(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Объединение прямоугольников. §5.2 п. 3: «`bbox` фрагмента = объединение
    `bbox` его блоков»."""
    if not boxes:
        raise ValueError("объединять нечего: пустой список прямоугольников")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def is_scanned(text_chars: int, *, has_images: bool) -> bool:
    """Страница — скан. §5.2 п. 2.

    Оба условия обязательны. Страница без текста и **без** изображений — это
    пустая страница, и гнать её в vision незачем: расход без результата.
    """
    return text_chars < SCAN_TEXT_THRESHOLD and has_images


@dataclass(frozen=True)
class HeadingPlan:
    """Разметка заголовков: кому какой заголовок и какие блоки — сами заголовки."""

    titles: list[str | None]
    """Для каждого блока — заголовок, под который он попадает, или `None`."""
    heading_at: frozenset[int]
    """Индексы блоков, которые **сами являются** заголовками.

    Нужны отдельно, потому что такой блок — разметка, а не содержание: его
    текст уходит в `locator.heading` блоков под ним, а фрагментом он не
    становится. Иначе заголовок попал бы в материал дважды — отдельным
    фрагментом в две строки и ключом локатора, — и §7.1 собрал бы из него
    факт.
    """


def analyze_headings(blocks: Sequence[Block]) -> HeadingPlan:
    """Заголовки — эвристика §5.2 п. 5 по кеглю.

    §5.2 п. 5 прямо: «На проверенном материале оглавления нет, то есть
    **штатным путём является эвристика, а не исключение**». Поэтому она здесь
    и покрыта тестами, а путь через `doc.get_toc()` — в адаптере.

    Блок считается заголовком, если его кегль **строго** больше медианного по
    документу. Страница, где все кегли равны, заголовков не даёт: медиана равна
    любому из них, и иначе каждый абзац стал бы заголовком.

    Сам блок-заголовок получает `None`: он не содержание, а разметка, и его
    текст уходит в локаторы блоков под ним.
    """
    if not blocks:
        return HeadingPlan(titles=[], heading_at=frozenset())

    median = statistics.median(block.size for block in blocks)
    current: str | None = None
    titles: list[str | None] = []
    heading_at: set[int] = set()
    for index, block in enumerate(blocks):
        if block.size > median and block.text.strip():
            current = " ".join(block.text.split())
            titles.append(None)
            heading_at.add(index)
            continue
        titles.append(current)
    return HeadingPlan(titles=titles, heading_at=frozenset(heading_at))


def glue(
    blocks: Sequence[Block],
    *,
    count_tokens: Callable[[str], int],
    plan: HeadingPlan | None = None,
) -> list[Piece]:
    """Склеивает блоки во фрагменты 600–900 токенов. §5.2 п. 3.

    **Фрагмент не пересекает границу страницы.** Это не оптимизация: локатор
    §3.4 хранит ровно один номер страницы, и склейка через разворот дала бы
    ссылку на источник, указывающую не туда.

    Заголовок фрагмента — заголовок его первого блока. Фрагмент, собранный из
    блоков под разными заголовками, не бывает: новый заголовок начинает новый
    фрагмент. Иначе `locator.heading` называл бы не тот раздел, а §7.1 собрал
    бы по нему неверную секцию.

    Блоки-заголовки пропускаются: их текст живёт в `locator.heading`.

    `count_tokens` передаётся параметром: считать должен тот же счётчик, что
    у оценки §6.3 (`core/llm/estimation.py::tokens_in`), иначе `fragments.tokens`
    и уточнённая оценка разойдутся на величину расхождения счётчиков.
    """
    headings = plan if plan is not None else analyze_headings(blocks)
    if len(headings.titles) != len(blocks):
        raise ValueError("заголовков должно быть столько же, сколько блоков")

    pieces: list[Piece] = []
    batch: list[Block] = []
    batch_heading: str | None = None
    batch_tokens = 0

    def flush() -> None:
        nonlocal batch, batch_tokens, batch_heading
        if not batch:
            return
        text = "\n\n".join(block.text.strip() for block in batch if block.text.strip())
        if text:
            pieces.append(
                Piece(
                    page=batch[0].page,
                    text=text,
                    bbox=union_bbox([block.bbox for block in batch]),
                    tokens=count_tokens(text),
                    heading=batch_heading,
                )
            )
        batch = []
        batch_tokens = 0
        batch_heading = None

    for index, (block, heading) in enumerate(zip(blocks, headings.titles, strict=True)):
        if not block.text.strip() or index in headings.heading_at:
            # Блок-заголовок фрагментом не становится: его текст уже уехал в
            # `locator.heading` блоков под ним.
            continue
        tokens = count_tokens(block.text)
        if tokens > MAX_FRAGMENT_TOKENS:
            # §5.2 п. 3 (v3.16): неделимый блок уходит во фрагмент целиком.
            # Запись в журнал — то, что отличает названное последствие от
            # молчаливого превышения границы.
            log.info("block_exceeds_max_tokens", page=block.page, tokens=tokens)

        breaks_page = bool(batch) and block.page != batch[0].page
        breaks_heading = bool(batch) and heading != batch_heading
        breaks_size = batch_tokens >= MIN_FRAGMENT_TOKENS or (
            batch_tokens + tokens > MAX_FRAGMENT_TOKENS
        )
        if breaks_page or breaks_heading or breaks_size:
            flush()

        if not batch:
            batch_heading = heading
        batch.append(block)
        batch_tokens += tokens

    flush()
    return pieces


def keep_pictures(pictures: Sequence[Picture], *, min_area_ratio: float) -> list[KeptPicture]:
    """Решает, какие изображения сохранить. §5.2 п. 4.

    Правило порога и правило группы действуют вместе:

    - изображение **вне** набора сохраняется, если его площадь больше
      `min_area_ratio` (§30.3, `MIN_IMAGE_AREA_RATIO` = 0.02);
    - изображения одной страницы с подписями «Рис. N», образующие
      **непрерывную** нумерацию, сохраняются целиком — независимо от площади
      каждого — и получают общий `group_id`.

    Непрерывность нумерации и есть признак набора: §5.2 приводит замер, где
    на странице «Рис. 1…10» восемь картинок из десяти не прошли порог, и
    разорванный набор бесполезен. Разрыв в номерах начинает новый набор:
    «Рис. 1, 2, 3» и «Рис. 7, 8» на одной странице — две разные группы, и
    общий `group_id` у них означал бы, что `match_image` соберёт задание из
    двух несвязанных иллюстраций.
    """
    numbered: dict[int, list[tuple[int, Picture]]] = {}
    singles: list[Picture] = []

    for picture in pictures:
        match = FIGURE_CAPTION.match(picture.caption or "")
        if match is None:
            singles.append(picture)
            continue
        numbered.setdefault(picture.page, []).append((int(match.group(1)), picture))

    kept: list[KeptPicture] = []

    for page, items in numbered.items():
        items.sort(key=lambda pair: pair[0])
        run: list[tuple[int, Picture]] = []
        for number, picture in items:
            if run and number != run[-1][0] + 1:
                kept.extend(_as_group(page, run, min_area_ratio))
                run = []
            run.append((number, picture))
        kept.extend(_as_group(page, run, min_area_ratio))

    kept.extend(
        KeptPicture(picture=picture, group_id=None)
        for picture in singles
        if picture.area_ratio > min_area_ratio
    )
    return kept


def _as_group(
    page: int, run: Sequence[tuple[int, Picture]], min_area_ratio: float
) -> list[KeptPicture]:
    """Один непрерывный набор «Рис. N» или одиночная подписанная картинка.

    Набор из одной картинки группой не является: `group_id` называет набор, и
    на одиночной подписи он означал бы набор из одного элемента, который
    `match_image` собрать не может (§9.2 требует минимум двух пар).
    """
    if not run:
        return []
    if len(run) == 1:
        picture = run[0][1]
        if picture.area_ratio > min_area_ratio:
            return [KeptPicture(picture=picture, group_id=None)]
        return []

    group_id = f"page{page}-fig{run[0][0]}"
    return [KeptPicture(picture=picture, group_id=group_id) for _, picture in run]
