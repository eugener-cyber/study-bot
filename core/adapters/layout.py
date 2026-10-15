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

FIGURE_CAPTION = re.compile(r"рис\.?\s*(\d+)(?!\s*\))", re.IGNORECASE)
r"""Подпись пронумерованного набора §5.2 п. 4.

Форма «Рис. N» взята из спецификации; точка необязательна, регистр не важен.
Другие формы (`Фото 1`, `Схема 2`) сюда не попадают намеренно: §5.2 называет
одну, а догадываться за спецификацию — §40.

**Поиск по всему блоку, а не с его начала, и с исключением «Рис. N)».** В PR A
стояло `^рис\.?\s*N` — замер по корпусу 49 файлов показал два дефекта такой
формы. Ложное срабатывание: блок «рис. 1) женского платья» — обрывок фразы, а
не подпись, и он единственный давал «подпись на странице без изображений».
Потеря: блок «грейфера Рис.3. Модель захвата» — настоящая подпись, склеенная
разбором с концом предыдущей строки, — не опознавалась, и третья картинка
набора оставалась вне группы (стр. 29 `tasks-tech-11-prak`).

Замер четырёх вариантов (подписей / файлов / ложных / привязанных картинок):
`^рис N` — 27/11/1/39; `^рис N[.-—:]` — 23/10/0/—; `рис N[.-—:]` где угодно —
24/10/0/—; **этот вариант вместе с `CAPTION_MAX_CHARS` — 27/10/0/40**. Только
он не теряет законных подписей вида «Рис. 1» и «Рис. 2 Фронтон Большого
театра», где после номера нет разделителя.

Следствие формы названо прямо: короткий блок «см. рис. 2 выше» станет
подписью. От этого защищает только исключение `рис N)`, и на корпусе ложных
срабатываний не осталось — но правило опознаёт форму, а не смысл.
"""

CAPTION_MAX_CHARS: Final = 120
"""Предел длины блока-подписи. §5.2 п. 4 о длине молчит — это замер.

Поиск «Рис. N» по всему блоку без границы длины превращал в подписи абзацы,
упоминающие рисунок: замер по корпусу — **10** таких блоков, самый длинный
**2634** символа (`tasks-tech-11-model-kddt-final-24-25.pdf`, стр. 7). Способ
подсчёта: блоки, которые проходят `FIGURE_CAPTION`, но длиннее этой границы.
Граница отсекает их все.

Числа замера: самая длинная подпись корпуса — 57 символов («Рис. 1. Виды
графических элементов для выполнения задания»), медиана 35, подписей 27 в 10
файлах на 49 файлах корпуса. Предел взят с запасом вдвое от максимума, а не
вплотную: подпись длиннее 57 символов встретиться может, абзац короче 120 —
вряд ли.
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
    xref: int = 0
    """Идентификатор объекта изображения внутри PDF.

    По нему идёт дедупликация: одно и то же изображение размещается в
    документе многократно — замер по корпусу 49 файлов даёт 86 лишних
    размещений в 10 файлах, максимум 31 копия (белая подложка страницы, доля
    площади 0.69). «Та же картинка» по определению PDF — это один `xref`, а не
    сходство пикселей, поэтому хеш содержимого не нужен: он дал бы тот же
    ответ дороже.
    """


@dataclass(frozen=True)
class Caption:
    """Подпись «Рис. N» на странице. §5.2 п. 4.

    Отдельная структура, а не поле `Picture`: подпись существует до того, как
    известно, каким изображениям она принадлежит, и одна подпись может
    принадлежать нескольким сразу — это и есть второй признак набора.
    Подпись без изображений тоже бывает (замер корпуса: такие есть) и обязана
    не ломать правило.
    """

    page: int
    number: int
    bbox: tuple[float, float, float, float]
    text: str


@dataclass(frozen=True)
class KeptPicture:
    """Изображение, которое сохраняется, и его группа."""

    picture: Picture
    group_id: str | None
    """Непустой у изображений пронумерованного набора «Рис. N» (§3.4, v3.15)."""
    caption: Caption | None = None
    """Подпись, которой изображение досталось, если она нашлась.

    Нужна дальше адаптеру: текст подписи уходит в `media_assets.caption` (§3),
    и пересчитывать привязку второй раз незачем.
    """


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


def captions_of(blocks: Sequence[Block]) -> list[Caption]:
    """Подписи «Рис. N» среди текстовых блоков страницы. §5.2 п. 4.

    Блок длиннее `CAPTION_MAX_CHARS` подписью не считается: замер в докстроке
    константы. Номер берётся первым вхождением — «Рис. 1» в блоке, который
    начинается словами «Рис. 1. Виды…», это и есть номер подписи.
    """
    out: list[Caption] = []
    for block in blocks:
        text = block.text.strip()
        if len(text) > CAPTION_MAX_CHARS:
            continue
        match = FIGURE_CAPTION.search(text)
        if match is None:
            continue
        out.append(Caption(page=block.page, number=int(match.group(1)), bbox=block.bbox, text=text))
    return out


def assign_captions(
    pictures: Sequence[Picture], captions: Sequence[Caption]
) -> dict[int, list[Picture]]:
    """Какая подпись каким изображениям принадлежит. Ключ — индекс в `captions`.

    Правило выведено замером по корпусу 49 файлов, а не из спецификации: §5.2
    п. 4 о связи подписи с изображением молчит. Изображение достаётся подписи,
    которая

    1. пересекается с ним **по горизонтали**;
    2. начинается **ниже верха** изображения;
    3. ближайшая по `|подпись.y0 − низ изображения|` **среди подходящих по
       пунктам 1–2**.

    Пункт 3 сформулирован именно так не из педантизма: прочтение «ближайшая
    среди всех подписей страницы» даёт 34 привязки вместо 40 и ломает страницу
    с рядом подписей в три столбца — там изображение уходит к подписи соседней
    колонки (замер: стр. 27 `tasks-tech-10-prak` даёт `{3: 1}` вместо
    `{1: 1, 2: 1, 3: 1}`).

    Чего в правиле **нет** — порога расстояния. Первая редакция плана ставила
    20 pt «под изображением»; замер показал, что это неверно дважды: зазоры по
    корпусу от −31.5 до +30.2 pt, медиана **−6.3**, то есть в 14 случаях из 26
    подпись начинается выше низа своей картинки, и порог 20 pt покрыл бы 8 из
    26. Пункт 2 сравнивает с **верхом** изображения, и порог не нужен.

    Итог замера этим правилом: 27 подписей в 10 файлах, 40 привязанных
    изображений, ни одной подписи без своих картинок.
    """
    out: dict[int, list[Picture]] = {index: [] for index, _ in enumerate(captions)}
    for picture in pictures:
        candidates = [
            (index, caption)
            for index, caption in enumerate(captions)
            if caption.page == picture.page
            and caption.bbox[0] < picture.bbox[2]
            and caption.bbox[2] > picture.bbox[0]
            and caption.bbox[1] > picture.bbox[1]
        ]
        if not candidates:
            continue
        index, _ = min(candidates, key=lambda item: abs(item[1].bbox[1] - picture.bbox[3]))
        out[index].append(picture)
    return out


def keep_pictures(
    pictures: Sequence[Picture],
    captions: Sequence[Caption] = (),
    *,
    min_area_ratio: float,
) -> list[KeptPicture]:
    """Решает, какие изображения сохранить и что из них набор. §5.2 п. 4.

    Правило порога и правило набора действуют вместе:

    - изображение **вне** набора сохраняется, если его площадь больше
      `min_area_ratio` (§30.3, `MIN_IMAGE_AREA_RATIO` = 0.02);
    - изображения набора сохраняются целиком, независимо от площади каждого, и
      получают общий `group_id`.

    **У набора два признака**, и оба нужны (ТЗ v3.17, решение Архитектора
    2026-10-15):

    - **ряд подписей** — «Рис. 1», «Рис. 2», «Рис. 3» с непрерывной нумерацией
      на одной странице; их изображения сливаются в один набор. Замер: 6
      страниц корпуса;
    - **гроздь под одной подписью** — у одной подписи несколько своих
      изображений. Замер по корпусу: **пять** таких подписей, способ подсчёта
      — подписи, которым `assign_captions` отдал ≥2 изображения. Самая крупная
      — страница, которую §5.2 приводит собственным примером (десять элементов
      под подписью «Рис. 1», нумерованных голыми числами); остальные четыре
      дают по две картинки.

    Одного первого признака недостаточно: он не собирает пример из самого §5.2.
    Одного второго — тоже: он разваливает шесть страниц корпуса на одиночные
    наборы. Приоритет между ними не нужен — ряд **сливает** подписи, гроздь
    берёт свои, и на странице с обеими формами (`tasks-tech-9-tvor`, стр. 2:
    `{1: 2, 2: 1}`) выходит один набор из трёх изображений.

    Разрыв нумерации начинает новый набор: «Рис. 1, 2, 3» и «Рис. 7, 8» на
    одной странице — две разные группы, и общий `group_id` означал бы, что
    `match_image` соберёт задание из двух несвязанных иллюстраций.
    """
    assigned = assign_captions(pictures, captions)
    owned = {id(picture) for group in assigned.values() for picture in group}

    kept: list[KeptPicture] = []
    for page_captions in _by_page(captions):
        for run in _runs(page_captions):
            members = [picture for index, _ in run for picture in assigned[index]]
            if not members:
                continue
            first = run[0][1]
            if len(members) == 1:
                # Набор из одной картинки набором не является: `group_id`
                # называет набор, и на одиночной подписи он означал бы набор
                # из одного элемента, который `match_image` собрать не может
                # (§9.2 требует минимум двух пар).
                if members[0].area_ratio > min_area_ratio:
                    kept.append(KeptPicture(picture=members[0], group_id=None, caption=first))
                continue
            group_id = f"page{first.page}-fig{first.number}"
            owners = {id(picture): caption for index, caption in run for picture in assigned[index]}
            kept.extend(
                KeptPicture(
                    picture=picture,
                    group_id=group_id,
                    caption=owners.get(id(picture)),
                )
                for picture in members
            )

    kept.extend(
        KeptPicture(picture=picture, group_id=None)
        for picture in pictures
        if id(picture) not in owned and picture.area_ratio > min_area_ratio
    )
    return kept


def _by_page(captions: Sequence[Caption]) -> list[list[tuple[int, Caption]]]:
    """Подписи, сгруппированные по странице, с сохранением индексов."""
    pages: dict[int, list[tuple[int, Caption]]] = {}
    for index, caption in enumerate(captions):
        pages.setdefault(caption.page, []).append((index, caption))
    return list(pages.values())


def _runs(page_captions: Sequence[tuple[int, Caption]]) -> list[list[tuple[int, Caption]]]:
    """Непрерывные ряды номеров внутри одной страницы.

    Дубль номера («Рис. 2» дважды на странице) ряд не разрывает и не
    удваивает: номера берутся по возрастанию, и повтор просто попадает в тот
    же ряд. На корпусе таких страниц нет ни одной — поведение названо, чтобы
    не выяснялось на первом же материале, где они появятся.
    """
    ordered = sorted(page_captions, key=lambda item: item[1].number)
    runs: list[list[tuple[int, Caption]]] = []
    for item in ordered:
        if runs and item[1].number - runs[-1][-1][1].number <= 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs
