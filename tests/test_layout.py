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
    MAX_FRAGMENT_TOKENS,
    SCAN_TEXT_THRESHOLD,
    Block,
    Picture,
    analyze_headings,
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


def _picture(page: int, ratio: float, caption: str | None = None, *, y: float = 0.0) -> Picture:
    return Picture(
        page=page,
        bbox=(72.0, y, 200.0, y + 100.0),
        area_ratio=ratio,
        caption=caption,
    )


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
    result = keep_pictures([_picture(1, ratio)], min_area_ratio=MIN_AREA_RATIO)

    assert bool(result) is kept, why


def test_numbered_set_survives_below_the_threshold() -> None:
    """§5.2 п. 4: набор «Рис. N» сохраняется целиком независимо от площади.

    Замер §5.2: из двадцати изображений реального задания порог прошли восемь,
    и отброшенными оказались ровно наборы, ради которых существует
    `match_image`.
    """
    pictures = [
        _picture(7, 0.009, "Рис. 1. Прямая строчка", y=0.0),
        _picture(7, 0.012, "Рис. 2. Зигзаг", y=120.0),
        _picture(7, 0.008, "Рис. 3. Оверлок", y=240.0),
    ]

    kept = keep_pictures(pictures, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 3, "набор разорван порогом площади"
    assert {item.group_id for item in kept} == {"page7-fig1"}


def test_a_gap_in_numbering_starts_a_second_group() -> None:
    """Два набора на странице получают разные `group_id`.

    Общий `group_id` означал бы, что `match_image` собёрет задание из двух
    несвязанных иллюстраций.
    """
    pictures = [
        _picture(7, 0.009, "Рис. 1", y=0.0),
        _picture(7, 0.009, "Рис. 2", y=100.0),
        _picture(7, 0.009, "Рис. 7", y=200.0),
        _picture(7, 0.009, "Рис. 8", y=300.0),
    ]

    kept = keep_pictures(pictures, min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 4
    assert {item.group_id for item in kept} == {"page7-fig1", "page7-fig7"}


def test_single_captioned_picture_is_not_a_group() -> None:
    """Набор из одного элемента `match_image` собрать не может.

    §9.2 требует от соотнесения минимум двух пар, поэтому `group_id` на
    одиночной подписи означал бы набор, которым нельзя воспользоваться. Порог
    площади к такой картинке применяется как к любой одиночной.
    """
    kept = keep_pictures([_picture(3, 0.05, "Рис. 4. Выкройка")], min_area_ratio=MIN_AREA_RATIO)

    assert len(kept) == 1
    assert kept[0].group_id is None


def test_sets_on_different_pages_do_not_merge() -> None:
    pictures = [
        _picture(2, 0.009, "Рис. 1", y=0.0),
        _picture(2, 0.009, "Рис. 2", y=100.0),
        _picture(5, 0.009, "Рис. 1", y=0.0),
        _picture(5, 0.009, "Рис. 2", y=100.0),
    ]

    kept = keep_pictures(pictures, min_area_ratio=MIN_AREA_RATIO)

    assert {item.group_id for item in kept} == {"page2-fig1", "page5-fig1"}


def test_caption_in_another_form_is_not_a_set() -> None:
    """Распознаётся ровно форма §5.2 — «Рис. N».

    «Фото 1» и «Схема 2» спецификация не называет, и догадываться за неё
    §40 запрещает: такие картинки проходят по общему порогу площади.
    """
    pictures = [
        _picture(1, 0.009, "Фото 1. Ткань", y=0.0),
        _picture(1, 0.009, "Схема 2. Шов", y=100.0),
    ]

    assert keep_pictures(pictures, min_area_ratio=MIN_AREA_RATIO) == []


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
