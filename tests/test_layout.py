"""Раскладка страницы: склейка, заголовки, группы изображений. ТЗ §5.2.

Все проверки — на простых структурах, без PDF на диске: нормы §5.2 нужно
гонять десятком раскладок, и каждая через сконструированный файл стоила бы
зависимости от версии библиотеки.

Нормы стоят **литералами** там, где они нормы: 900 токенов, порог площади
0.02, сто символов детекта скана. Урок `STATE.md` про `TRUNCATE_AT`: тест,
выраженный только через импортированную константу, зелен при любом её
значении.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from core.adapters.layout import (
    CAPTION_MAX_CHARS,
    MAX_FRAGMENT_TOKENS,
    SCAN_TEXT_THRESHOLD,
    Block,
    Caption,
    Picture,
    analyze_headings,
    assign_captions,
    captions_of,
    glue,
    is_scanned,
    keep_pictures,
    union_bbox,
)
from core.llm.estimation import tokens_in

MAX_TOKENS = 900
"""§5.2 п. 3. Литерал намеренно."""

MIN_AREA_RATIO = 0.02
"""§30.3, `MIN_IMAGE_AREA_RATIO`. Литерал намеренно."""

SCAN_CHARS = 100
"""§5.2 п. 2. Литерал намеренно."""


def test_spec_constants_match_the_module() -> None:
    """Литералы теста совпадают с константами модуля и §30.3."""
    from core.config import Settings

    assert MAX_FRAGMENT_TOKENS == MAX_TOKENS
    assert SCAN_TEXT_THRESHOLD == SCAN_CHARS
    assert Settings.model_fields["MIN_IMAGE_AREA_RATIO"].default == MIN_AREA_RATIO


def _block(page: int, text: str, *, size: float = 10.0, y: float = 0.0) -> Block:
    return Block(page=page, text=text, bbox=(72.0, y, 523.0, y + 20.0), size=size)


# --- Склейка §5.2 п. 3 -----------------------------------------------------


def test_fragment_never_crosses_a_page_boundary() -> None:
    """§5.2 п. 3: локатор хранит один номер страницы.

    Блоки обеих страниц подаются **одним** вызовом: при постраничном вызове
    утверждение тождественно истинно, и тест не отличал бы верную реализацию
    от склейки через разворот.
    """
    blocks = [
        _block(1, "Первая страница, абзац один."),
        _block(1, "Первая страница, абзац два."),
        _block(2, "Вторая страница, абзац один."),
    ]

    pieces = glue(blocks, count_tokens=tokens_in)

    assert len(pieces) == 2
    assert [piece.page for piece in pieces] == [1, 2]
    assert "Вторая" not in pieces[0].text
    assert "Первая" not in pieces[1].text


def test_upper_bound_is_enforced_and_a_long_fragment_actually_appears() -> None:
    """Верхняя граница 900 токенов соблюдается — и граница достигается.

    Положительный ассерт обязателен: «ни один фрагмент не больше 900»
    выполняется и на пустом выходе, и на выходе из односложных фрагментов. Но
    и он должен требовать именно **подхода к границе**: ассерт «больше 300»
    был истинным и тогда, когда склейки не было вовсе и каждый блок фикстуры
    становился отдельным фрагментом.

    Размер абзаца подобран замером под эту форму ассерта: 43 повтора дают 295
    токенов в блоке, три блока склеиваются в 884 — внутри 850–900. При 44
    повторах первый же блок переваливает нижнюю границу раньше, и фрагменты
    выходят по 603 токена, то есть полоса 850–900 этой фикстурой уже не
    достигается.
    """
    paragraph = "Длинный абзац про раскрой ткани. " * 43
    blocks = [_block(1, paragraph, y=float(i * 30)) for i in range(6)]

    pieces = glue(blocks, count_tokens=tokens_in)

    assert pieces, "склейка не дала ни одного фрагмента"
    assert max(piece.tokens for piece in pieces) <= MAX_TOKENS
    assert (
        850 <= max(piece.tokens for piece in pieces) <= MAX_TOKENS
    ), "граница не достигнута — проверка «не больше 900» ничего не доказывает"


def test_an_indivisible_block_goes_whole_and_is_named_in_the_journal() -> None:
    """§5.2 п. 3 (v3.16): абзац длиннее 900 токенов не режется.

    Решение Архитектора 2026-10-14 по замечанию ревью PR #42: граница
    ограничивает склейку, а неделимый блок уходит во фрагмент целиком — резать
    его по предложениям нельзя, иначе два фрагмента получат один и тот же
    `bbox`. До правки граница нарушалась молча, и докстрока константы
    утверждала обратное тому, что делал код.

    Запись в журнал — вторая половина решения: без неё превышение остаётся
    невидимым. Замер этой фикстуры: 1023 токена в блоке.
    """
    paragraph = "Неделимый абзац про раскрой ткани без единого разрыва. " * 90
    blocks = [_block(7, paragraph)]

    with capture_logs() as logs:
        pieces = glue(blocks, count_tokens=tokens_in)

    assert len(pieces) == 1, "блок разрезан, хотя §5.2 п. 3 это запрещает"
    assert pieces[0].text == paragraph.strip(), "текст абзаца изменён"
    assert pieces[0].tokens > MAX_TOKENS
    assert pieces[0].bbox == blocks[0].bbox

    exceeded = [entry for entry in logs if entry["event"] == "block_exceeds_max_tokens"]
    assert len(exceeded) == 1, "превышение границы не названо в журнале"
    assert exceeded[0]["page"] == 7
    assert exceeded[0]["tokens"] == pieces[0].tokens


def test_a_block_within_the_bound_is_not_named_in_the_journal() -> None:
    """Обратное направление: иначе запись стояла бы на каждом блоке."""
    with capture_logs() as logs:
        glue([_block(1, "Короткий абзац.")], count_tokens=tokens_in)

    assert not [entry for entry in logs if entry["event"] == "block_exceeds_max_tokens"]


def test_sparse_page_gives_a_fragment_below_the_lower_bound() -> None:
    """Нижняя граница 600 токенов необязательна. §5.2 п. 3 дословно."""
    pieces = glue([_block(1, "Короткая строка.")], count_tokens=tokens_in)

    assert len(pieces) == 1
    assert pieces[0].tokens < 600


def test_bbox_of_a_fragment_is_the_union_of_its_blocks() -> None:
    blocks = [
        Block(page=1, text="верх", bbox=(100.0, 50.0, 200.0, 70.0), size=10.0),
        Block(page=1, text="низ", bbox=(80.0, 90.0, 300.0, 120.0), size=10.0),
    ]

    pieces = glue(blocks, count_tokens=tokens_in)

    assert len(pieces) == 1
    assert pieces[0].bbox == (80.0, 50.0, 300.0, 120.0)


def test_union_of_nothing_is_an_error_not_a_zero_box() -> None:
    """Нулевой прямоугольник выглядел бы как валидный локатор §3.4."""
    with pytest.raises(ValueError, match="пустой список"):
        union_bbox([])


# --- Заголовки §5.2 п. 5 ---------------------------------------------------


def test_heading_is_a_block_with_a_larger_font() -> None:
    """Эвристика §5.2 п. 5: кегль строго выше медианного."""
    blocks = [
        _block(1, "Раскрой ткани", size=18.0),
        _block(1, "Первый абзац раздела."),
        _block(1, "Второй абзац раздела."),
    ]

    plan = analyze_headings(blocks)

    assert plan.titles == [None, "Раскрой ткани", "Раскрой ткани"]


def test_uniform_font_gives_no_headings() -> None:
    """Страница с одинаковым кеглем заголовков не даёт.

    Иначе медиана равна любому блоку, и каждый абзац стал бы заголовком — то
    есть `locator.heading` заполнился бы мусором на всём материале без
    структуры.
    """
    blocks = [_block(1, f"Абзац {i}.") for i in range(4)]

    assert analyze_headings(blocks).titles == [None, None, None, None]


def test_second_heading_replaces_the_first() -> None:
    blocks = [
        _block(1, "Раскрой", size=18.0),
        _block(1, "Про раскрой."),
        _block(1, "Пошив", size=18.0),
        _block(1, "Про пошив."),
    ]

    assert analyze_headings(blocks).titles == [None, "Раскрой", None, "Пошив"]


def test_fragment_does_not_mix_two_headings() -> None:
    """Фрагмент под двумя заголовками сделал бы `locator.heading` ложным.

    §7.1 собирает секции по этому ключу, и фрагмент, отнесённый не к своему
    разделу, попал бы в чужую секцию вместе с фактами.
    """
    blocks = [
        _block(1, "Раскрой", size=18.0),
        _block(1, "Про раскрой."),
        _block(1, "Пошив", size=18.0),
        _block(1, "Про пошив."),
    ]

    pieces = glue(blocks, count_tokens=tokens_in)

    assert [piece.heading for piece in pieces] == ["Раскрой", "Пошив"]
    assert pieces[0].text == "Про раскрой."
    assert pieces[1].text == "Про пошив."


# --- Детект скана §5.2 п. 2 ------------------------------------------------


@pytest.mark.parametrize(
    ("chars", "has_images", "expected", "why"),
    [
        (SCAN_CHARS - 1, True, True, "99 символов при картинке — скан"),
        (SCAN_CHARS, True, False, "ровно 100 — уже не скан, граница строгая"),
        (SCAN_CHARS + 1, True, False, "101 символ — текстовый слой есть"),
        (SCAN_CHARS - 1, False, False, "пустая страница без картинок — не скан"),
        (0, False, False, "совсем пустая — vision незачем"),
    ],
)
def test_scan_detection_boundaries(chars: int, has_images: bool, expected: bool, why: str) -> None:
    """§5.2 п. 2 по обе стороны порога и по обоим условиям.

    Вторая половина — «при непустом изображении» — существенна: страница без
    текста и без картинок это пустая страница, и vision на ней означал бы
    расход без результата.
    """
    assert is_scanned(chars, has_images=has_images) is expected, why


# --- Изображения и правило группы §5.2 п. 4 --------------------------------


def _pic(
    page: int,
    ratio: float = 0.03,
    *,
    x0: float = 72.0,
    y: float = 0.0,
    width: float = 128.0,
    height: float = 100.0,
    xref: int = 1,
) -> Picture:
    return Picture(
        page=page,
        bbox=(x0, y, x0 + width, y + height),
        area_ratio=ratio,
        xref=xref,
    )


def _cap(
    page: int,
    number: int,
    *,
    x0: float = 72.0,
    y: float = 110.0,
    width: float = 128.0,
    text: str | None = None,
) -> Caption:
    return Caption(
        page=page,
        number=number,
        bbox=(x0, y, x0 + width, y + 14.0),
        text=text if text is not None else f"Рис. {number}. Подпись",
    )


# --- опознание подписи ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "is_caption", "why"),
    [
        ("Рис. 1. Виды графических элементов", True, "каноническая форма §5.2"),
        ("Рис. 1", True, "после номера нет ничего — тоже подпись"),
        ("Рис. 2 Фронтон Большого театра", True, "разделителя после номера нет"),
        ("грейфера Рис.3. Модель захвата", True, "подпись склеена с концом строки разбором"),
        ("рис. 1) женского платья", False, "обрывок фразы, а не подпись"),
        ("Выберите элементы (рис.1).", False, "ссылка на рисунок внутри текста"),
        ("Фото 1. Ткань", False, "§5.2 называет форму «Рис. N», догадываться за неё — §40"),
    ],
)
def test_caption_form_is_recognised_by_measurement(text: str, is_caption: bool, why: str) -> None:
    """Форма подписи — замер по корпусу, а не догадка. §5.2 п. 4.

    Каждая строка взята из настоящих материалов: первые четыре правило обязано
    брать, остальные — отбрасывать. На форме из PR A (`^рис N`) краснеют две:
    «рис. 1)» становилась подписью, а склеенный блок — не становился.
    """
    blocks = [Block(page=1, text=text, bbox=(72.0, 100.0, 400.0, 114.0), size=10.0)]

    assert bool(captions_of(blocks)) is is_caption, why


def test_a_long_block_mentioning_a_figure_is_not_a_caption() -> None:
    """Граница длины §5.2 п. 4 (замер): без неё подписями становятся абзацы.

    Замер по корпусу: 10 блоков длиной до 2037 символов упоминают «рис. N» в
    тексте. Самая длинная настоящая подпись — 57 символов.
    """
    long_text = "Заметать складку по линии середины, см. рис. 5 выше. " * 5
    assert len(long_text) > CAPTION_MAX_CHARS

    blocks = [Block(page=1, text=long_text, bbox=(72.0, 100.0, 500.0, 300.0), size=10.0)]

    assert captions_of(blocks) == []


# --- привязка подписи к изображениям --------------------------------------


def test_cluster_under_one_caption_is_one_set() -> None:
    """Второй признак набора: гроздь под одной подписью. §5.2 п. 4 v3.17.

    Координаты — замер со стр. 10 файла живого прогона, той самой, которую
    §5.2 приводит собственным примером: десять элементов пронумерованы голыми
    числами, подпись одна. Правило PR A («ряд подписей») этот набор не
    собирало ни при каком опознании подписи.
    """
    boxes = [
        (162.1, 61.7, 282.9, 184.2),
        (385.0, 61.7, 520.8, 184.1),
        (162.4, 191.0, 282.5, 311.4),
        (374.6, 191.0, 531.0, 317.4),
        (150.9, 324.3, 293.9, 456.1),
        (141.1, 462.9, 303.9, 590.8),
        (386.8, 463.0, 518.9, 596.8),
        (148.2, 603.6, 296.8, 729.0),
        (388.2, 603.6, 517.5, 734.4),
        (368.7, 326.0, 537.0, 447.3),
    ]
    pictures = [
        Picture(page=10, bbox=box, area_ratio=0.03, xref=index + 1)
        for index, box in enumerate(boxes)
    ]
    captions = [
        Caption(
            page=10,
            number=1,
            bbox=(140.9, 742.0, 480.0, 756.0),
            text="Рис. 1. Виды графических элементов для выполнения задания",
        )
    ]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 10, "гроздь собралась не целиком"
    assert {item.group_id for item in kept} == {"page10-fig1"}


def test_a_row_of_captions_in_columns_is_one_set() -> None:
    """Первый признак: ряд подписей с непрерывной нумерацией сливается в набор.

    Форма со стр. 27 `tasks-tech-10-prak`: три подписи стоят **в один ряд** в
    трёх столбцах, у каждой своя картинка. Различаются они по горизонтали —
    по вертикали они почти совпадают (замер: y0 = 384, 385, 391).
    """
    pictures = [_pic(27, x0=x, y=100.0, height=200.0) for x in (60.0, 220.0, 380.0)]
    captions = [
        _cap(27, number, x0=x, y=310.0) for number, x in ((1, 60.0), (2, 220.0), (3, 380.0))
    ]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 3
    assert {item.group_id for item in kept} == {"page27-fig1"}


def test_nearest_caption_is_chosen_among_candidates_not_globally() -> None:
    """Пункт 3 правила: ближайшая **среди подходящих по пунктам 1–2**.

    Координаты — замер стр. 27 `tasks-tech-10-prak-kddt-final-21-22.pdf`, и
    взяты они именно оттуда не для красоты: на этой геометрии глобальное
    прочтение правила даёт `{3: 3}` — все три картинки уходят к третьей
    подписи, потому что она ближе по вертикали к низу каждой. Первая редакция
    теста стояла на выдуманных координатах, где своя подпись была ближайшей и
    при глобальном прочтении, то есть тест проходил **и при сломанном
    правиле**; нашло ревью PR #47 мутацией реализации.

    Числа: подписи `Рис.1` y0=384.2, `Рис.2` y0=385.0, `Рис.3` y0=391.0;
    картинки с низом 388.9, 394.9 и 398.9. Правильное прочтение даёт каждой
    картинке свою подпись — различает их **горизонталь**, а не вертикаль.
    """
    pictures = [
        Picture(page=27, bbox=(71.0, 271.1, 188.8, 388.9), area_ratio=0.0277, xref=1),
        Picture(page=27, bbox=(200.4, 271.1, 338.1, 394.9), area_ratio=0.0340, xref=2),
        Picture(page=27, bbox=(347.2, 282.6, 555.3, 398.9), area_ratio=0.0483, xref=3),
    ]
    captions = [
        Caption(page=27, number=1, bbox=(72.7, 384.2, 192.5, 398.2), text="Рис.1. Захват двух"),
        Caption(page=27, number=2, bbox=(212.7, 385.0, 341.5, 399.0), text="Рис.2. Захват с двумя"),
        Caption(page=27, number=3, bbox=(357.1, 391.0, 558.8, 405.0), text="Рис.3. Модель захвата"),
    ]

    assigned = assign_captions(pictures, captions)

    assert {index: len(group) for index, group in assigned.items()} == {
        0: 1,
        1: 1,
        2: 1,
    }, "картинки ушли к чужой подписи — правило прочитано как «глобально ближайшая»"
    assert assigned[0] == [pictures[0]]
    assert assigned[1] == [pictures[1]]
    assert assigned[2] == [pictures[2]]


def test_a_picture_below_its_caption_is_not_attached() -> None:
    """Пункт 2: подпись обязана начинаться ниже верха изображения.

    Иначе подпись забирала бы картинку, стоящую под ней, — то есть
    принадлежащую следующей подписи.
    """
    picture = _pic(1, y=200.0)
    caption = _cap(1, 1, y=100.0)

    assert assign_captions([picture], [caption]) == {0: []}


def test_a_caption_without_pictures_does_not_break_the_rule() -> None:
    """Замер корпуса: такие подписи есть. Правило обязано их переживать."""
    captions = [_cap(9, 1)]

    assert assign_captions([], captions) == {0: []}
    assert keep_pictures([], captions, min_area_ratio=MIN_AREA_RATIO) == []


# --- порог и наборы §5.2 п. 4, §30.3 --------------------------------------


@pytest.mark.parametrize(
    ("ratio", "kept", "why"),
    [
        (0.0199, False, "ниже порога 0.02"),
        (0.0201, True, "выше порога 0.02"),
    ],
)
def test_threshold_is_checked_at_its_own_boundary(ratio: float, kept: bool, why: str) -> None:
    """Порог §30.3 проверяется вплотную, а не на 0.9% и 4%.

    Первая редакция этого теста брала 0.9% и 4% — такие значения прошли бы и
    при пороге 0.01, и при 0.03, то есть норму §30.3 тест не держал.
    """
    result = keep_pictures([_pic(1, ratio)], min_area_ratio=MIN_AREA_RATIO)

    assert bool(result) is kept, why


def test_numbered_set_survives_below_the_threshold() -> None:
    """§5.2 п. 4: набор «Рис. N» сохраняется целиком независимо от площади.

    Замер §5.2: из двадцати изображений реального задания порог прошли восемь,
    и отброшенными оказались ровно наборы, ради которых существует
    `match_image`.
    """
    pictures = [_pic(7, 0.009, y=float(index) * 120.0) for index in range(3)]
    captions = [_cap(7, index + 1, y=float(index) * 120.0 + 105.0) for index in range(3)]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 3, "набор разорван порогом площади"
    assert {item.group_id for item in kept} == {"page7-fig1"}


def test_a_cluster_below_the_threshold_survives_whole() -> None:
    """Гроздь под одной подписью освобождена от порога §30.3, как и ряд.

    План обещал этот тест отдельно от ряда подписей, и не зря: прежняя
    проверка «набор ниже порога» стояла на **ряде** из трёх подписей, то есть
    второй признак набора порогом не проверялся вовсе. Мутация «порог
    применяется к грозди» не роняла ни одного теста — нашло ревью PR #47.

    Девять картинок по 0.01 под одной подписью: без правила их отбросил бы
    порог 0.02 целиком, и `match_image` остался бы без материала — ровно то,
    против чего §5.2 п. 4 и вводился.
    """
    pictures = [_pic(6, 0.01, x0=60.0 + index * 50.0, y=100.0) for index in range(9)]
    captions = [_cap(6, 1, x0=60.0, y=210.0, width=500.0)]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 9, "гроздь ниже порога отброшена"
    assert {item.group_id for item in kept} == {"page6-fig1"}


def test_a_gap_in_numbering_starts_a_second_group() -> None:
    """Два набора на странице получают разные `group_id`.

    Общий `group_id` означал бы, что `match_image` соберёт задание из двух
    несвязанных иллюстраций.
    """
    pictures = [
        _pic(7, 0.009, x0=60.0, y=100.0),
        _pic(7, 0.009, x0=220.0, y=100.0),
        _pic(7, 0.009, x0=60.0, y=400.0),
        _pic(7, 0.009, x0=220.0, y=400.0),
    ]
    captions = [
        _cap(7, 1, x0=60.0, y=210.0),
        _cap(7, 2, x0=220.0, y=210.0),
        _cap(7, 7, x0=60.0, y=510.0),
        _cap(7, 8, x0=220.0, y=510.0),
    ]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 4
    assert {item.group_id for item in kept} == {"page7-fig1", "page7-fig7"}


def test_duplicate_numbers_on_a_page_do_not_split_the_run() -> None:
    """Дубль номера ряд не разрывает. На корпусе таких страниц нет ни одной.

    Поведение названо здесь, а не выясняется на первом материале, где они
    появятся: §5.2 о дублях молчит.
    """
    pictures = [_pic(4, 0.009, x0=60.0, y=100.0), _pic(4, 0.009, x0=220.0, y=100.0)]
    captions = [_cap(4, 2, x0=60.0, y=210.0), _cap(4, 2, x0=220.0, y=210.0)]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert {item.group_id for item in kept} == {"page4-fig2"}


def test_single_captioned_picture_is_not_a_group() -> None:
    """Набор из одного элемента `match_image` собрать не может.

    §9.2 требует от соотнесения минимум двух пар, поэтому `group_id` на
    одиночной подписи означал бы набор, которым нельзя воспользоваться. Порог
    площади к такой картинке применяется как к любой одиночной.
    """
    kept = keep_pictures([_pic(3, 0.05)], [_cap(3, 4)], min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 1
    assert kept[0].group_id is None


def test_a_single_asset_gets_its_own_caption_not_the_first_of_the_run() -> None:
    """Подпись одиночного ассета — его собственная. Ревью PR #48.

    Ряд может начинаться подписью без картинок: «Рис. 1» без иллюстрации и
    «Рис. 2» со своей. Первая редакция писала в `KeptPicture.caption` первую
    подпись ряда, и картинка под «Рис. 2» получала текст «Рис. 1» — а он
    уезжает в `media_assets.caption` и дальше в описание vision (C2).

    На корпусе таких страниц сегодня ноль, поэтому тест стоит на
    сконструированной раскладке: ошибка нашлась чтением кода, а не данными, и
    закрепить её нужно до того, как она проявится на материале.
    """
    picture = _pic(3, 0.05, x0=220.0, y=100.0)
    captions = [
        _cap(3, 1, x0=60.0, y=210.0, text="Рис. 1. Без картинки"),
        _cap(3, 2, x0=220.0, y=210.0, text="Рис. 2. Своя"),
    ]

    kept = keep_pictures([picture], captions, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 1
    assert kept[0].caption is not None
    assert kept[0].caption.text == "Рис. 2. Своя", "ассет получил чужую подпись"
    assert kept[0].group_id is None, "набор из одной картинки набором не считается"


def test_sets_on_different_pages_do_not_merge() -> None:
    pictures = [
        _pic(2, 0.009, x0=60.0, y=100.0),
        _pic(2, 0.009, x0=220.0, y=100.0),
        _pic(5, 0.009, x0=60.0, y=100.0),
        _pic(5, 0.009, x0=220.0, y=100.0),
    ]
    captions = [
        _cap(2, 1, x0=60.0, y=210.0),
        _cap(2, 2, x0=220.0, y=210.0),
        _cap(5, 1, x0=60.0, y=210.0),
        _cap(5, 2, x0=220.0, y=210.0),
    ]

    kept = keep_pictures(pictures, captions, min_area_ratio=MIN_AREA_RATIO)

    assert {item.group_id for item in kept} == {"page2-fig1", "page5-fig1"}


def test_pictures_without_captions_go_by_the_threshold_alone() -> None:
    """Изображение вне набора живёт по порогу §30.3, и ничего больше."""
    pictures = [_pic(1, 0.009), _pic(1, 0.05, y=300.0)]

    kept = keep_pictures(pictures, [], min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 1
    assert kept[0].picture.area_ratio == 0.05
    assert kept[0].group_id is None


def test_heading_block_does_not_become_a_fragment() -> None:
    """Заголовок попадает в локатор, а не в материал вторым фрагментом.

    Первая редакция склейки делала из блока-заголовка отдельный фрагмент в две
    строки с пустым `heading` — то есть заголовок попадал в материал дважды, и
    §7.1 собрал бы из него факт.
    """
    blocks = [
        _block(1, "Раскрой ткани", size=18.0),
        _block(1, "Мел размечает ткань по лицевой стороне."),
    ]

    pieces = glue(blocks, count_tokens=tokens_in)

    assert len(pieces) == 1
    assert pieces[0].heading == "Раскрой ткани"
    assert "Раскрой ткани" not in pieces[0].text
