"""Ассеты в базе: стадия `extract`. ТЗ §3, §4.2, §4.3, §5.2 п. 4, §6.3.

Все проверки идут **через `run_stage`**, а не вызовом обработчика напрямую:
постоянный путь и перенос файлов существуют только в границе конвейера, и
тест, вызывающий обработчик, проверял бы временный каталог — то есть прошёл бы
и при записи в базу временного пути. Требование ревью Плана C1b.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.base import ExtractedAsset, ExtractedFragment, SourceMeta
from core.adapters.stub import StubAdapter
from core.db.models import Fragment, Material, MediaAsset, User
from core.ingest.handlers import ASSETS_DIRNAME, ASSETS_ORIGIN_KEY
from core.ingest.handlers import extract as extract_stage
from core.ingest.pipeline import StageContext, run_stage
from core.ingest.stages import EXTRACT
from core.storage import MaterialStorage

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _asset(
    name: str, *, page: int = 1, ratio: float = 0.03, group: str | None = None
) -> ExtractedAsset:
    return ExtractedAsset(
        path=name,
        mime="image/png",
        width=10,
        height=20,
        caption=f"Рис. 1. Подпись {name}" if group else None,
        page=page,
        area_ratio=ratio,
        group_id=group,
    )


def _adapter(assets: list[ExtractedAsset] | None = None) -> StubAdapter:
    return StubAdapter(
        meta=SourceMeta(pages=1, text_chars=200),
        fragments=[
            ExtractedFragment(
                ord=1,
                kind="text",
                locator={"type": "web", "url": "stub://1", "quote": "цитата"},
                text="Утверждение.",
                quality=1.0,
                tokens=10,
            )
        ],
        assets=assets,
    )


async def _material(engine: AsyncEngine, tg_id: int) -> int:
    async with engine.begin() as connection:
        user_id = (
            await connection.execute(
                insert(User).values(tg_id=tg_id, created_at=NOW).returning(User.id)
            )
        ).scalar_one()
        return int(
            (
                await connection.execute(
                    insert(Material)
                    .values(
                        user_id=user_id,
                        kind="pdf",
                        status="processing",
                        mode="full",
                        completed_stages=[],
                        origin={"sha256": "a" * 64, "path": "./data/materials/x.pdf"},
                        progress={},
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


async def _run(
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    material_id: int,
    adapter: StubAdapter,
) -> bool:
    def ctx_factory(tmp_dir: Path) -> StageContext:
        return StageContext(
            material_id=material_id,
            tmp_dir=tmp_dir,
            adapter=adapter,
            source_path="stub://source",
            permanent_dir=storage.permanent_dir(material_id),
        )

    return await run_stage(sessions, storage, extract_stage, EXTRACT, ctx_factory, material_id)


async def _rows(
    engine: AsyncEngine, material_id: int
) -> list[tuple[int, str, str, int, int, str | None]]:
    async with engine.connect() as connection:
        return [
            tuple(row)  # type: ignore[misc]
            for row in (
                await connection.execute(
                    select(
                        MediaAsset.id,
                        MediaAsset.path,
                        MediaAsset.mime,
                        MediaAsset.w,
                        MediaAsset.h,
                        MediaAsset.caption,
                    )
                    .where(MediaAsset.material_id == material_id)
                    .order_by(MediaAsset.id)
                )
            ).all()
        ]


async def _origin(engine: AsyncEngine, material_id: int) -> dict[str, object]:
    async with engine.connect() as connection:
        value = (
            await connection.execute(select(Material.origin).where(Material.id == material_id))
        ).scalar_one()
    return dict(value or {})


# --- строки и файлы -------------------------------------------------------


async def test_assets_reach_the_database_and_the_file_lives_where_it_says(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Строка `media_assets` есть, и файл существует по пути **из базы**.

    Путь склеивается с корнем хранилища, а не сравнивается со строкой,
    которую тест сам же и собрал: иначе проверка свелась бы к равенству
    строки самой себе (требование ревью Плана C1b).
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9001)

    assert await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    rows = await _rows(engine, material_id)
    assert len(rows) == 1
    _, path, mime, width, height, caption = rows[0]
    assert (tmp_path / path).is_file(), f"файла нет по пути из базы: {path}"
    assert (tmp_path / path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "файл повреждён переносом"
    assert (mime, width, height) == ("image/png", 10, 20)
    assert caption is None, "подпись взялась не из ExtractedAsset"


async def test_path_is_relative_to_the_storage_root(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Форма пути зафиксирована: `<material_id>/assets/<имя>`.

    Решение плана C1b: ни абсолютный путь, ни относительный к рабочему
    каталогу. Воркер работает в контейнере с `WORKDIR /app`, харнесс — в
    каталоге разработчика, и оба склеивают значение со своим корнем.
    Абсолютный путь из контейнера на хосте не существует.
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9002)

    await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    path = (await _rows(engine, material_id))[0][1]
    assert path == f"{material_id}/{ASSETS_DIRNAME}/img001.png"
    assert not path.startswith("/"), "путь абсолютный"
    assert str(tmp_path) not in path, "в путь попало имя корня хранилища"


async def test_rerun_with_fewer_assets_leaves_no_stale_files(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Повторный запуск заменяет каталог ассетов целиком. §4.3.

    Это проверка механизма, выбранного вместо нового канала «стадия →
    конвейер»: `commit_tmp` удаляет существующую цель, и если цель — каталог,
    он удаляется со всем содержимым. Без подкаталога `assets/` файл
    `img002.png` остался бы на диске, а строки на него уже не было бы.
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9003)
    assets_dir = storage.permanent_dir(material_id) / ASSETS_DIRNAME

    await _run(
        sessions, storage, material_id, _adapter([_asset("img001.png"), _asset("img002.png")])
    )
    assert sorted(p.name for p in assets_dir.iterdir()) == ["img001.png", "img002.png"]

    # Стадия уже отмечена выполненной — повторный запуск идёт через сброс
    # отметки, как это делает кнопка «Обработать заново» (§4.1).
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=[])
        )

    await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    rows = await _rows(engine, material_id)
    assert len(rows) == 1, "строки прошлого запуска не удалены"
    assert sorted(p.name for p in assets_dir.iterdir()) == [
        "img001.png"
    ], "файл прошлого запуска остался на диске"
    assert (tmp_path / rows[0][1]).is_file()


async def test_rerun_replaces_the_origin_list(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """`origin.assets.items` заменяется, а не дописывается. Требование ревью."""
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9004)

    await _run(
        sessions, storage, material_id, _adapter([_asset("img001.png"), _asset("img002.png")])
    )
    first = await _origin(engine, material_id)
    assert len(first[ASSETS_ORIGIN_KEY]["items"]) == 2  # type: ignore[index]

    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=[])
        )
    await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    second = await _origin(engine, material_id)
    assert len(second[ASSETS_ORIGIN_KEY]["items"]) == 1, "перечень дописан, а не заменён"  # type: ignore[index]
    assert second["sha256"] == "a" * 64, "слияние стёрло ключ дубликата §4.4"


async def test_group_and_geometry_live_in_origin(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Страница, площадь и группа — в `origin`, колонок под них в §3 нет.

    Долг с владельцем: C2 переносит группу в локатор фрагмента (§3.4) и
    чистит ключ. Здесь проверяется, что до тех пор она не теряется.
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9005)
    assets = [
        _asset("img001.png", page=10, ratio=0.0295, group="page10-fig1"),
        _asset("img002.png", page=10, ratio=0.0332, group="page10-fig1"),
    ]

    await _run(sessions, storage, material_id, _adapter(assets))

    items = (await _origin(engine, material_id))[ASSETS_ORIGIN_KEY]["items"]  # type: ignore[index]
    assert {item["group_id"] for item in items} == {"page10-fig1"}
    assert {item["page"] for item in items} == {10}
    assert sorted(item["area_ratio"] for item in items) == [0.0295, 0.0332]
    rows = await _rows(engine, material_id)
    assert {item["asset_id"] for item in items} == {
        row[0] for row in rows
    }, "перечень в origin не связан со строками базы"


async def test_material_without_images_gets_no_rows_and_no_origin_key(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Текстовый материал не обрастает пустыми структурами.

    Отрицательная половина: иначе проверки выше прошли бы и у реализации,
    которая пишет ключ и каталог всегда.
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9006)

    assert await _run(sessions, storage, material_id, _adapter())

    assert await _rows(engine, material_id) == []
    assert ASSETS_ORIGIN_KEY not in await _origin(engine, material_id)


async def test_stage_failure_leaves_neither_rows_nor_files(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой **после** записи строк и файлов не оставляет половины стадии. §4.2.

    Сбой ставится в `merge_origin` — то есть после вставки `media_assets` и
    после записи файлов во временный каталог. Предусловие проверяется в самом
    тесте: без подмены те же данные доходят до базы и диска.
    """
    storage = MaterialStorage(tmp_path)

    # Предусловие: на том же материале и тех же ассетах стадия проходит.
    ok_id = await _material(engine, tg_id=9007)
    await _run(sessions, storage, ok_id, _adapter([_asset("img001.png")]))
    assert len(await _rows(engine, ok_id)) == 1
    assert (storage.permanent_dir(ok_id) / ASSETS_DIRNAME / "img001.png").is_file()

    from core.ingest import handlers as module

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("сбой после записи строк и файлов")

    monkeypatch.setattr(module, "merge_origin", boom)

    material_id = await _material(engine, tg_id=9008)
    with pytest.raises(RuntimeError, match="сбой после записи"):
        await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    assert await _rows(engine, material_id) == [], "строки остались после откака"
    assert not (
        storage.permanent_dir(material_id) / ASSETS_DIRNAME
    ).exists(), "файлы перенесены при сбойной стадии"
    async with engine.connect() as connection:
        stages = (
            await connection.execute(
                select(Material.completed_stages).where(Material.id == material_id)
            )
        ).scalar_one()
    assert EXTRACT.name not in list(stages or []), "стадия отмечена выполненной при сбое"


async def test_fragments_are_deleted_before_assets(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Порядок удаления обязателен по схеме, а не по состоянию данных.

    `fragments.asset_id` ссылается на `media_assets.id` — ограничение
    `fragments_asset_id_fkey` есть и в модели, и в миграции, и в базе.
    Удаление ассетов раньше фрагментов нарушило бы его на строках со ссылкой.

    Сам пакет ссылок не создаёт: `asset_id` не заполняется, потому что
    фрагментов-изображений здесь нет (решение Архитектора). Но ждать C2, чтобы
    проверить порядок, незачем — ссылка ставится **в тесте**, как её поставит
    C2. Без неё тест ничего не держал: перестановка удалений оставляла все
    проверки зелёными, и это найдено мутацией, а не предположено.
    """
    storage = MaterialStorage(tmp_path)
    material_id = await _material(engine, tg_id=9009)

    await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    asset_id = (await _rows(engine, material_id))[0][0]
    async with engine.begin() as connection:
        await connection.execute(
            Fragment.__table__.update()
            .where(Fragment.material_id == material_id)
            .values(asset_id=asset_id)
        )
        await connection.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(completed_stages=[])
        )

    # Повторный запуск: при обратном порядке удаления Postgres поднимет
    # нарушение `fragments_asset_id_fkey` — фрагмент ещё ссылается на ассет.
    await _run(sessions, storage, material_id, _adapter([_asset("img001.png")]))

    async with engine.connect() as connection:
        fragments = (
            await connection.execute(
                select(Fragment.id, Fragment.asset_id).where(Fragment.material_id == material_id)
            )
        ).all()
    assert len(fragments) == 1, "фрагменты не переписаны"
    assert fragments[0][1] is None, "новый фрагмент унаследовал ссылку прошлого запуска"
    assert len(await _rows(engine, material_id)) == 1
