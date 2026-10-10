"""Консольный харнесс разработчика. План реализации §1.4.

«Агент не видит Telegram и не может проверить работу вручную. Поэтому с первого
пакета существует консольный интерфейс, покрывающий весь цикл… Это не
вспомогательный инструмент, а основной способ разработки: почти все проверки
«Готово, если» выполняются через него».

Здесь живёт `seed` — прогон материала через конвейер целиком. Остальные цели
§1.4 появляются в своих пакетах: `dump-material` в WP-06, `session` и `answer`
в WP-05, `cite` в WP-11, `evals` в WP-07.

Запуск: `make seed FILE=путь/к/файлу.md`
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.adapters.base import ExtractedFragment, SourceMeta
from core.adapters.stub import StubAdapter
from core.config import load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.models import User
from core.db.session import session_scope
from core.ingest import handlers as h
from core.ingest.pipeline import StageHandler, process_material
from core.logging import configure_logging, get_logger
from core.services.materials import create_or_find, mark_ready, sha256_of
from core.storage import MaterialStorage

log = get_logger(__name__)

MATERIALS_ROOT = "./data/materials"
"""§30.1, том материалов."""


def fragments_from_text(path: Path) -> list[ExtractedFragment]:
    """Делит текстовый файл на фрагменты по пустым строкам.

    Это харнесс, а не адаптер: настоящие адаптеры — PDF в WP-06, аудио в
    WP-19. Деление по абзацам выбрано потому, что им же устроены конспекты,
    которые готовятся вручную для провайдера `manual` (§2.1, CR-H): один
    абзац источника — один фрагмент с разрешимой ссылкой.

    Локатор берётся варианта `web` из §3.4 — `url` и `quote`, — с `file://` в
    качестве URL. Новый вариант `Locator` здесь не вводится: §5.1 говорит, что
    новый тип источника это адаптер **плюс** вариант локатора, то есть правка
    §3.4, то есть change request. Для харнесса это не нужно.
    """
    raw = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    return [
        ExtractedFragment(
            ord=index,
            kind="text",
            locator={
                "type": "web",
                "url": f"file://{path.resolve()}",
                "quote": block[:60],
            },
            text=block,
            quality=1.0,
            tokens=max(len(block) // 5, 1),
        )
        for index, block in enumerate(blocks, start=1)
    ]


async def seed(file_path: str, tg_id: int | None = None) -> int:
    """Прогоняет файл через конвейер целиком и возвращает `material_id`.

    Пользователь берётся первый из белого списка, если не задан явно: §1.3
    закрывает доступ белым списком, и материал, созданный на посторонний id,
    не был бы виден никому.
    """
    settings = load_settings()
    path = Path(file_path)
    if not path.is_file():
        raise SystemExit(f"файл не найден: {file_path}")

    fragments = fragments_from_text(path)
    if not fragments:
        raise SystemExit(f"файл пуст или не разбился на абзацы: {file_path}")

    owner_tg_id = tg_id if tg_id is not None else settings.ALLOWED_USER_IDS[0]

    engine = build_engine(settings.DATABASE_URL)
    sessions = build_sessionmaker(engine)
    storage = MaterialStorage(MATERIALS_ROOT)

    try:
        async with session_scope(sessions) as session:
            user_id = await _ensure_user(session, owner_tg_id)
            material, duplicate = await create_or_find(
                session,
                user_id=user_id,
                kind="text",
                title=path.name,
                origin={
                    "sha256": sha256_of(path),
                    "orig_name": path.name,
                    "path": str(path.resolve()),
                },
            )
            material_id = material.id
            if duplicate:
                print(f"материал уже есть: id={material_id} — §4.4, копия не создаётся")

        adapter = StubAdapter(
            meta=SourceMeta(text_chars=len(path.read_text(encoding="utf-8"))),
            fragments=fragments,
        )
        await process_material(
            sessions,
            storage,
            _handlers(settings, sessions),
            material_id,
            adapter,
            str(path.resolve()),
        )

        async with session_scope(sessions) as session:
            await mark_ready(session, material_id)
    finally:
        await engine.dispose()

    print(f"готово: material_id={material_id}, фрагментов={len(fragments)}")
    return material_id


def _handlers(settings: object, sessions: object) -> dict[str, StageHandler]:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from core.config import Settings
    from core.llm.factory import build_client

    assert isinstance(settings, Settings)
    assert isinstance(sessions, async_sessionmaker)
    return h.build_handlers(build_client(settings, sessions), settings)


async def _ensure_user(session: object, tg_id: int) -> int:
    """Пользователь по `tg_id`, создавая при необходимости.

    Отдельно от `accept_consent`: тот фиксирует согласие, а харнесс согласия
    не получает и подделывать его не должен. `consent_at` остаётся пустым, и
    по нему видно, что пользователь создан инструментом, а не человеком.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    assert isinstance(session, AsyncSession)
    existing = (
        await session.execute(select(User.id).where(User.tg_id == tg_id))
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    statement = (
        pg_insert(User)
        .values(tg_id=tg_id)
        .on_conflict_do_nothing(index_elements=["tg_id"])
        .returning(User.id)
    )
    created = (await session.execute(statement)).scalar_one_or_none()
    if created is not None:
        return int(created)
    return int((await session.execute(select(User.id).where(User.tg_id == tg_id))).scalar_one())


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="harness", description="Консольный харнесс")
    sub = parser.add_subparsers(dest="command", required=True)

    seed_parser = sub.add_parser("seed", help="прогнать файл через конвейер")
    seed_parser.add_argument("file", help="путь к текстовому файлу")
    seed_parser.add_argument(
        "--tg-id", type=int, default=None, help="Telegram id владельца материала"
    )

    args = parser.parse_args()
    if args.command == "seed":
        asyncio.run(seed(args.file, args.tg_id))


if __name__ == "__main__":
    main()
