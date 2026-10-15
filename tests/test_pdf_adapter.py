"""PDF-адаптер: `probe`, фрагменты, локаторы, рендер. ТЗ §5.1, §5.2, §3.4, §6.3.

Фикстуры строятся кодом (`tests/pdf_fixtures.py`), бинарников в репозитории
нет. Номер страницы закрепляется **против содержания** страницы, а не против
обратного преобразования: пара «прибавить единицу при разборе, вычесть при
рендере» самосогласована, и тест через неё остаётся зелёным при симметричной
ошибке в обеих конвертациях — урок `STATE.md` про фейк из той же константы.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from core.adapters.base import ExtractedAsset, IngestCtx
from core.adapters.factory import build_adapter, kind_for_path, known_kind_for_path
from core.adapters.pdf import (
    MAX_IMAGE_SIDE,
    RENDER_DPI,
    VISION_PENDING_KEY,
    NotAPdfError,
    PdfAdapter,
)
from core.config import Settings
from core.ingest.budget import estimate_from_probe
from tests.conftest import ENV_STUB
from tests.pdf_fixtures import (
    PAGE_ONE_MARKER,
    PAGE_TWO_MARKER,
    cyrillic_font,
    figures_pdf,
    lying_page_count_pdf,
    mixed_pdf,
    oversized_image_pdf,
    scanned_pdf,
    text_pdf,
    zero_page_pdf,
)

A4_HEIGHT_AT_150_DPI = 1755
"""Высота страницы A4 (842 pt) при 150 dpi. Замер, не расчёт: 842 / 72 * 150
даёт 1754.17, а PyMuPDF округляет до 1755."""


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, **ENV_STUB)  # type: ignore[arg-type]


@pytest.fixture
def adapter(settings: Settings) -> PdfAdapter:
    return PdfAdapter(settings=settings)


def _ctx(tmp_path: Path) -> IngestCtx:
    return IngestCtx(material_id=1, tmp_dir=str(tmp_path))


async def _fragments(adapter: PdfAdapter, path: Path, tmp_path: Path) -> list[object]:
    return [item async for item in adapter.extract(str(path), _ctx(tmp_path))]


# --- probe §5.1, §6.3 ------------------------------------------------------


async def test_probe_counts_pages_and_text_without_calling_the_model(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """§5.1: `probe` дешёвый. Модель не вызывается — клиента у адаптера нет вовсе."""
    meta = await adapter.probe(str(text_pdf(tmp_path / "t.pdf")))

    assert meta.pages == 2
    assert meta.text_chars is not None and meta.text_chars > 400
    assert meta.vision_pages_ratio == 0.0
    assert meta.extra[VISION_PENDING_KEY] == []


async def test_probe_reports_the_share_of_scanned_pages(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """§6.3: доля страниц под vision — главный множитель оценки.

    Считается **до** `extract`, потому что для скана `extract` это vision-вызов
    на каждую страницу, то есть основная статья расхода.
    """
    meta = await adapter.probe(str(mixed_pdf(tmp_path / "m.pdf")))

    assert meta.pages == 2
    assert meta.vision_pages_ratio == 0.5
    assert meta.extra[VISION_PENDING_KEY] == [2]


async def test_probe_of_a_full_scan_says_every_page_needs_vision(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    meta = await adapter.probe(str(scanned_pdf(tmp_path / "s.pdf")))

    assert meta.vision_pages_ratio == 1.0
    assert meta.extra[VISION_PENDING_KEY] == [1]


# --- фрагменты и локаторы §5.2, §3.4 ---------------------------------------


async def test_page_number_in_locator_is_bound_to_page_content(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Номер страницы закреплён содержанием, а не обратным преобразованием.

    PyMuPDF нумерует с нуля, локатор §3.4 хранит человеческий номер с единицы
    (ADR-0007). Привязка к маркеру в тексте ловит сдвиг на единицу в разборе;
    привязка к рендеру — в рендере. Пары конвертаций, согласованной между
    собой, этому тесту недостаточно.
    """
    fragments = await _fragments(adapter, text_pdf(tmp_path / "t.pdf"), tmp_path)

    by_marker = {
        PAGE_ONE_MARKER: None,
        PAGE_TWO_MARKER: None,
    }
    for fragment in fragments:
        for marker in by_marker:
            if marker in fragment.text:  # type: ignore[attr-defined]
                by_marker[marker] = fragment  # type: ignore[assignment]

    assert by_marker[PAGE_ONE_MARKER] is not None, "фрагмент первой страницы не найден"
    assert by_marker[PAGE_TWO_MARKER] is not None, "фрагмент второй страницы не найден"
    assert by_marker[PAGE_ONE_MARKER].locator["page"] == 1  # type: ignore[union-attr]
    assert by_marker[PAGE_TWO_MARKER].locator["page"] == 2  # type: ignore[union-attr]


async def test_locator_carries_heading_and_bbox(adapter: PdfAdapter, tmp_path: Path) -> None:
    """§3.4 (v3.15): `heading` — необязательный ключ варианта `pdf`.

    Заголовок берётся эвристикой §5.2 п. 5 по кеглю: в фикстуре он 18 против
    10 у основного текста.
    """
    fragments = await _fragments(adapter, text_pdf(tmp_path / "t.pdf"), tmp_path)

    first = next(f for f in fragments if PAGE_ONE_MARKER in f.text)  # type: ignore[attr-defined]
    assert first.locator["type"] == "pdf"  # type: ignore[attr-defined]
    assert first.locator["heading"] == "Fabric cutting"  # type: ignore[attr-defined]
    assert len(first.locator["bbox"]) == 4  # type: ignore[attr-defined]
    assert first.quality == 1.0, "§5.2 п. 6: текстовый слой даёт quality = 1.0"  # type: ignore[attr-defined]


async def test_tokens_are_counted_by_the_same_counter_as_the_estimate(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """`fragments.tokens` питает уточнённую оценку §6.3.

    Второй счётчик разошёлся бы с первым, и коридор §6.4 поехал бы на величину
    расхождения. Поэтому сверка идёт с `estimation.tokens_in`, а не с
    приблизительным «символы делить на пять».
    """
    from core.llm.estimation import tokens_in

    fragments = await _fragments(adapter, text_pdf(tmp_path / "t.pdf"), tmp_path)

    for fragment in fragments:
        assert fragment.tokens == tokens_in(fragment.text)  # type: ignore[attr-defined]


async def test_scanned_page_is_skipped_but_not_lost(adapter: PdfAdapter, tmp_path: Path) -> None:
    """Смешанный файл: текстовая страница обработана, скан назван в `probe`.

    §6.3 требует, чтобы последствия были видимыми: страница, выпавшая из
    материала молча, означала бы `ready` с неизвестным содержанием. Обработка
    таких страниц приезжает в PR B пакета.
    """
    path = mixed_pdf(tmp_path / "m.pdf")

    fragments = await _fragments(adapter, path, tmp_path)
    meta = await adapter.probe(str(path))

    assert fragments, "текстовая страница тоже выпала"
    assert {f.locator["page"] for f in fragments} == {1}  # type: ignore[attr-defined]
    assert meta.extra[VISION_PENDING_KEY] == [2], "пропущенная страница не названа"


async def test_full_scan_gives_no_fragments_so_the_stage_will_fail(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Файл целиком из сканов до PR B фрагментов не даёт — и это верно.

    Стадия упадёт на существующей проверке `core/ingest/handlers.py` («адаптер
    не дал ни одного фрагмента»), материал уйдёт в `failed` с внятной
    причиной. Молчаливый `ready` без содержания был бы хуже.
    """
    fragments = await _fragments(adapter, scanned_pdf(tmp_path / "s.pdf"), tmp_path)

    assert fragments == []


async def test_vision_ratio_from_probe_reaches_the_budget_estimate(
    adapter: PdfAdapter, settings: Settings, tmp_path: Path
) -> None:
    """§6.3: доля страниц под vision доходит до оценки, а не остаётся в мете.

    Половинки этого свойства проверяются и по отдельности — `probe` отдаёт
    0.5, а `tests/test_budget.py` считает цену по собранной руками `SourceMeta`
    — но между ними нет ни одной проверки, и именно стык обещал план.
    Оценка берётся из `core/ingest/budget.py`, а не пересчитывается тестом:
    пересчёт повторил бы ошибку расчёта, если она там есть.

    Сравниваются два файла одинаковой длины в страницах — смешанный и
    целиком текстовый, — потому что абсолютное число ничего не доказывает:
    §6.3 требует, чтобы страница под vision стоила **дороже** текстовой.
    """
    mixed = await adapter.probe(str(mixed_pdf(tmp_path / "m.pdf")))
    text = await adapter.probe(str(text_pdf(tmp_path / "t.pdf")))

    assert mixed.pages == text.pages == 2, "сравнение имеет смысл при равном числе страниц"
    assert mixed.vision_pages_ratio > text.vision_pages_ratio == 0.0

    mixed_cost = estimate_from_probe(mixed, settings)
    text_cost = estimate_from_probe(text, settings)

    assert mixed_cost.est_tokens > text_cost.est_tokens

    # Замер при этих настройках: 6050 токенов против 3950. Сравнивается
    # `est_tokens`, а не `est_cost`: провайдер в тестовых настройках —
    # `manual` (§2.1), у него прайс нулевой, и цена обеих оценок равна нулю,
    # то есть ассерт по цене не различал бы ничего.
    assert mixed_cost.est_cost == text_cost.est_cost == 0


# --- рендер §5.2 п. 2, §23 -------------------------------------------------


async def test_render_matches_an_independently_computed_page(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Рендер по локатору даёт ту же страницу, что прямой вызов PyMuPDF.

    Ожидание вычислено **в тесте** и другим путём: это и отличает проверку от
    самосогласованной пары конвертаций. Сравниваются `samples`, а не PNG:
    контейнер зависит от версии кодировщика, а пиксели — нет.
    """
    path = text_pdf(tmp_path / "t.pdf")

    rendered = adapter.render_page(str(path), 2)

    document = pymupdf.open(str(path))
    try:
        expected = document.load_page(1).get_pixmap(dpi=RENDER_DPI)
    finally:
        document.close()

    actual = pymupdf.Pixmap(rendered)
    assert actual.samples == expected.samples
    assert actual.height == A4_HEIGHT_AT_150_DPI


async def test_render_of_another_page_differs(adapter: PdfAdapter, tmp_path: Path) -> None:
    """Иначе предыдущий тест прошёл бы и при игнорировании номера страницы."""
    path = text_pdf(tmp_path / "t.pdf")

    first = hashlib.sha256(adapter.render_page(str(path), 1)).hexdigest()
    second = hashlib.sha256(adapter.render_page(str(path), 2)).hexdigest()

    assert first != second


async def test_highlight_changes_pixels_inside_the_box_and_not_outside(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Критерий приёмки WP-06: локатор превращается в подсвеченную страницу.

    Проверяется сравнением рендеров, а не непустотой PNG: ассерт «файл не
    пустой» не может упасть. Внутри `bbox` пиксели обязаны отличаться, вне —
    совпадать.
    """
    path = text_pdf(tmp_path / "t.pdf")
    bbox = (72.0, 120.0, 300.0, 200.0)

    plain = pymupdf.Pixmap(adapter.render_page(str(path), 1))
    marked = pymupdf.Pixmap(adapter.render_highlighted(str(path), 1, bbox))

    scale = RENDER_DPI / 72.0
    # Точка берётся **на линии рамки** — середина верхней границы `bbox`, — а
    # не внутри области: рамка рисуется по контуру, и пиксель в середине
    # прямоугольника остаётся белым при верной реализации тоже.
    on_border = (int((bbox[0] + bbox[2]) / 2 * scale), int(bbox[1] * scale))
    far_outside = (int(bbox[2] * scale) + 40, int(bbox[3] * scale) + 40)

    assert plain.pixel(*on_border) != marked.pixel(*on_border), "рамка не нарисована"
    assert plain.pixel(*far_outside) == marked.pixel(*far_outside), "задето вне bbox"


async def test_every_locator_turns_into_a_highlighted_page(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Критерий приёмки целиком: **каждый** локатор, а не первый.

    После PR C в выборке появятся локаторы `image` без `page` — ветвь по типу
    добавится там же. Пропуск по типу запрещён: ветвь либо проверяет, либо
    падает, иначе тест тихо перестанет покрывать половину материала.
    """
    path = text_pdf(tmp_path / "t.pdf")
    fragments = await _fragments(adapter, path, tmp_path)

    assert fragments
    for fragment in fragments:
        locator = fragment.locator  # type: ignore[attr-defined]
        assert locator["type"] == "pdf", f"неизвестный тип локатора: {locator}"
        image = pymupdf.Pixmap(
            adapter.render_highlighted(str(path), locator["page"], locator["bbox"])
        )
        assert image.height == A4_HEIGHT_AT_150_DPI


async def test_render_refuses_a_page_number_outside_the_document(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Номер вне документа — отказ, а не чужая страница.

    `0` и `-1` проверяются отдельно от `3`: без проверки первые два **не
    падают**, потому что PyMuPDF адресует отрицательные индексы с конца.
    Замер до правки на этой же фикстуре: `page = 0` отдавал байт-в-байт
    страницу 2, `page = -1` — страницу 1.
    """
    path = text_pdf(tmp_path / "t.pdf")
    last = hashlib.sha256(adapter.render_page(str(path), 2)).hexdigest()

    for bad in (0, -1, 3):
        with pytest.raises(ValueError, match="вне документа"):
            adapter.render_page(str(path), bad)
        with pytest.raises(ValueError, match="вне документа"):
            adapter.render_highlighted(str(path), bad, (0.0, 0.0, 10.0, 10.0))

    # Положительная половина: отказ не достигнут тем, что рендер сломан вовсе.
    assert hashlib.sha256(adapter.render_page(str(path), 2)).hexdigest() == last


# --- изображения §5.2 п. 4, §5.3 ------------------------------------------


async def _assets(adapter: PdfAdapter, path: Path, tmp_path: Path) -> list[ExtractedAsset]:
    return [item async for item in adapter.assets(str(path), _ctx(tmp_path))]


@pytest.mark.skipif(
    cyrillic_font() is None,
    reason="нет системного шрифта с кириллицей: подпись «Рис. N» не набрать",
)
async def test_assets_keep_the_numbered_set_and_drop_nothing_else(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """§5.2 п. 4: набор «Рис. N» сохраняется целиком, одиночное — по порогу.

    Фикстура: три мелких изображения с подписями «Рис. 1, 2, 3» на первой
    странице и одно крупное без подписи на второй. Проверяется и состав, и
    группа: без группы три мелких отбросил бы порог §30.3.

    Это единственный тест, которому нужна **кириллица** в подписи: правило
    §5.2 п. 4 опознаёт форму «Рис. N», а встроенные шрифты PyMuPDF кириллицы
    не несут (замер: «Рис. 1» извлекается как «···. 1»). Берётся системный
    шрифт, а не файл в репозитории (ADR-0007 решение 7).

    Пропуск оформлен `skipif` с видимой причиной, и этого **недостаточно**:
    пропущенный тест выглядит зелёным прогоном. Поэтому шрифт ставится в CI
    отдельным шагом (`.github/workflows/ci.yml`) — иначе проверка правила на
    настоящем разборе PDF не выполнялась бы там вовсе.
    """
    path = figures_pdf(tmp_path / "f.pdf", caption_prefix="Рис.")

    assets = await _assets(adapter, path, tmp_path)

    groups = {asset.group_id for asset in assets}
    assert len(assets) == 4, "состав ассетов изменился"
    assert groups == {"page1-fig1", None}, "набор не собрался или собрался не тот"
    in_group = [asset for asset in assets if asset.group_id == "page1-fig1"]
    assert len(in_group) == 3
    assert all(
        asset.area_ratio is not None and asset.area_ratio < 0.02 for asset in in_group
    ), "набор проверяется не на тех картинках: они обязаны быть ниже порога"
    assert all(asset.caption is not None for asset in in_group)


async def test_repeated_placement_of_one_image_gives_one_asset(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Дедупликация по `xref`. Замер корпуса: 86 лишних размещений, максимум 31.

    В фикстуре крупное изображение второй страницы размещено дважды, и оба
    размещения выше порога §30.3 — то есть без дедупликации ассетов стало бы
    два, и в C2 одна и та же картинка ушла бы в vision двумя вызовами.
    Подписи здесь не нужны: проверка не зависит от наличия кириллического
    шрифта.
    """
    path = figures_pdf(tmp_path / "f.pdf")

    assets = await _assets(adapter, path, tmp_path)

    assert len(assets) == 1, f"дедупликация не сработала: {[a.page for a in assets]}"
    assert assets[0].page == 2
    files = [item.name for item in tmp_path.iterdir() if item.suffix != ".pdf"]
    assert len(files) == 1, f"на диске лишние файлы: {sorted(files)}"


async def test_scanned_page_gives_no_assets(adapter: PdfAdapter, tmp_path: Path) -> None:
    """Страница-скан — предмет vision как страница (§5.2 п. 2), не иллюстрация.

    Замер: изображение скана занимает всю площадь страницы при нуле символов
    текста, то есть порог §30.3 его пропускает. Без этой ветви материал-скан
    превратился бы в набор «иллюстраций», каждая из которых — целая страница.
    """
    scan = scanned_pdf(tmp_path / "scan.pdf")

    assert await _assets(adapter, scan, tmp_path) == []

    # Положительная половина: тот же адаптер на файле с настоящими
    # иллюстрациями ассеты даёт — иначе проверка прошла бы и у реализации,
    # которая не отдаёт ничего никогда.
    assert await _assets(adapter, figures_pdf(tmp_path / "f.pdf"), tmp_path)


async def test_long_side_is_resized_to_the_limit(adapter: PdfAdapter, tmp_path: Path) -> None:
    """§5.3: ресайз по длинной стороне до 1568 px, и только вниз."""
    big = await _assets(adapter, oversized_image_pdf(tmp_path / "big.pdf"), tmp_path)
    small = await _assets(adapter, oversized_image_pdf(tmp_path / "small.pdf", side=800), tmp_path)

    assert len(big) == 1 and len(small) == 1
    assert max(big[0].width, big[0].height) == MAX_IMAGE_SIDE
    assert big[0].width / big[0].height == pytest.approx(2.0, abs=0.01), "пропорции не сохранены"
    assert max(small[0].width, small[0].height) == 800, "мелкую картинку растянули"


async def test_assets_are_written_only_into_the_stage_tmp_dir(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """§4.2: адаптер пишет только во временную директорию стадии.

    Постоянный путь выбирает стадия (PR C1b). Проверяется не словом докстроки:
    путь каждого ассета обязан лежать внутри `ctx.tmp_dir`, и файл обязан
    существовать.
    """
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    path = figures_pdf(tmp_path / "f.pdf")

    assets = [item async for item in adapter.assets(str(path), _ctx(stage_dir))]

    assert assets
    for asset in assets:
        file = Path(asset.path)
        assert file.parent == stage_dir, f"ассет записан вне tmp: {file}"
        assert file.exists() and file.stat().st_size > 0
        assert asset.mime.startswith("image/")


async def test_a_document_without_images_yields_no_assets(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Текстовый PDF ассетов не даёт — и это не ошибка."""
    assert await _assets(adapter, text_pdf(tmp_path / "t.pdf"), tmp_path) == []


# --- error case: не-PDF §5.1 -----------------------------------------------


async def test_probe_and_extract_refuse_a_file_that_is_not_a_pdf(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Материал, помеченный `pdf` по ошибке, отказывает, а не даёт мусор.

    До правки `pymupdf.open` определял формат по расширению: замер на обычном
    `notes.md` — документ из одной «страницы» и 11 символов текста, без
    исключения. То есть материал уходил дальше по конвейеру с одним
    бессмысленным фрагментом. Проверяются все четыре публичных входа — у
    `probe` и `extract` error-case не было ни одного.
    """
    not_pdf = tmp_path / "notes.md"
    not_pdf.write_text("Просто текстовый конспект, не PDF.\n", encoding="utf-8")

    with pytest.raises(NotAPdfError):
        await adapter.probe(str(not_pdf))
    with pytest.raises(NotAPdfError):
        await _fragments(adapter, not_pdf, tmp_path)
    with pytest.raises(NotAPdfError):
        adapter.render_page(str(not_pdf), 1)
    with pytest.raises(NotAPdfError):
        adapter.render_highlighted(str(not_pdf), 1, (0.0, 0.0, 10.0, 10.0))


async def test_no_open_failure_leaks_the_library_exception_type(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """Ни один отказ открытия не отдаёт тип PyMuPDF. §35, ADR-0007 решение 10.

    Пять входов, и до правки по замечанию ревью фазы 2 два из них уезжали
    наружу типом библиотеки: отсутствующий файл давал
    `pymupdf.FileNotFoundError` (потомок `RuntimeError`, **не** встроенного), а
    документ с неверным числом страниц — голый
    `RuntimeError: code=7: Invalid number of pages`, потому что `pymupdf.open`
    его проходит и падает уже `page_count`. Пятый вход — корректный PDF без
    страниц: он открывался молча, и `probe` отдавал `pages = 0`.

    Проверяется **модуль** типа исключения, а не только его имя:
    `pymupdf.FileNotFoundError` и встроенный `FileNotFoundError` называются
    одинаково, и ассерт по имени считал бы утечку закрытой.
    """
    real = text_pdf(tmp_path / "ok.pdf")

    not_pdf = tmp_path / "notes.md"
    not_pdf.write_text("Просто текстовый конспект, не PDF.\n", encoding="utf-8")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    missing = tmp_path / "нет-такого.pdf"

    for path, expected in (
        (not_pdf, NotAPdfError),
        (empty, NotAPdfError),
        (lying_page_count_pdf(tmp_path / "lying.pdf"), NotAPdfError),
        (zero_page_pdf(tmp_path / "zero.pdf"), NotAPdfError),
        (missing, FileNotFoundError),
    ):
        with pytest.raises(expected) as caught:
            await adapter.probe(str(path))
        assert not type(caught.value).__module__.startswith("pymupdf"), (
            f"{path.name}: наружу уехал {type(caught.value).__module__}."
            f"{type(caught.value).__name__}"
        )

    # Положительная половина: настоящий PDF по-прежнему читается, то есть
    # отказы получены не поломкой `_open`.
    assert (await adapter.probe(str(real))).pages == 2


async def test_the_refusal_does_not_leak_the_pymupdf_exception_type(
    adapter: PdfAdapter, tmp_path: Path
) -> None:
    """§35 и ADR-0007: тип библиотеки не уезжает из единственного модуля с ней.

    Иначе поймать этот случай мог бы только тот, кто сам импортирует PyMuPDF —
    то есть граница, которую держит `test_pymupdf_lives_in_one_module`, была бы
    обойдена через тип исключения. Причина при этом сохранена в `__cause__`.
    """
    not_pdf = tmp_path / "notes.md"
    not_pdf.write_text("Просто текстовый конспект, не PDF.\n", encoding="utf-8")

    with pytest.raises(NotAPdfError) as caught:
        await adapter.probe(str(not_pdf))

    assert not isinstance(caught.value, pymupdf.FileDataError)
    assert isinstance(caught.value.__cause__, pymupdf.FileDataError)


# --- фабрика §5.1 ----------------------------------------------------------


def test_factory_picks_the_adapter_by_kind(settings: Settings) -> None:
    assert isinstance(build_adapter("pdf", settings=settings), PdfAdapter)


def test_factory_returns_none_for_text_material(settings: Settings) -> None:
    """Текстовый материал разбирает харнесс, адаптера для него §5 не вводит.

    `None` вместо исключения: иначе прогон конспекта через `make seed` падал бы
    на выборе адаптера.
    """
    assert build_adapter("text", settings=settings) is None


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("a.pdf", "pdf"),
        ("A.PDF", "pdf"),
        ("conspect.md", "text"),
        ("notes.txt", "text"),
        ("photo-ab12cd34.jpg", "image"),
        ("scan.JPEG", "image"),
        ("page.png", "image"),
        ("shot.webp", "image"),
    ],
)
def test_kind_is_decided_by_extension(name: str, kind: str) -> None:
    assert kind_for_path(name) == kind


def test_unknown_extension_is_text_for_the_harness_and_none_for_the_chat() -> None:
    """Одна карта, две политики на неизвестном расширении — и обе названы.

    `make seed` читает как текст что угодно: это его назначение. Приём файла
    из чата обязан отказать внятно (`UNSUPPORTED_KIND`, §26), иначе `.docx`
    молча стал бы текстовым материалом.
    """
    assert kind_for_path("report.docx") == "text"
    assert known_kind_for_path("report.docx") is None
    assert known_kind_for_path("a.pdf") == "pdf"


def test_factory_refuses_a_kind_whose_adapter_has_not_arrived(settings: Settings) -> None:
    """Тип без адаптера — отказ с именем пакета, а не `None`.

    `None` означал бы «разбирает харнесс», и материал-изображение уходил бы в
    текстовую заглушку: ноль фрагментов и отказ стадии с причиной, из которой
    не видно, что тип просто не поддержан.
    """
    with pytest.raises(ValueError, match="WP-06 PR C"):
        build_adapter("image", settings=settings)
    with pytest.raises(ValueError, match="WP-19"):
        build_adapter("audio", settings=settings)
    with pytest.raises(ValueError, match="неизвестный тип"):
        build_adapter("хогвартс", settings=settings)
