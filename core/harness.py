"""Консольный харнесс разработчика. План реализации §1.4.

«Агент не видит Telegram и не может проверить работу вручную. Поэтому с первого
пакета существует консольный интерфейс, покрывающий весь цикл… Это не
вспомогательный инструмент, а основной способ разработки: почти все проверки
«Готово, если» выполняются через него».

Здесь живут `seed` — прогон материала через конвейер целиком — и пара
`session`/`answer`: сборка сеанса повторения и ответ на текущее задание.
Остальные цели §1.4 появляются в своих пакетах: `dump-material` в WP-06,
`cite` в WP-11, `evals` в WP-07.

Запуск:

```
make seed FILE=путь/к/файлу.md
make session
make answer OPTION=1
make answer IDK=1
```
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.adapters.base import ExtractedFragment, SourceMeta
from core.adapters.factory import build_adapter, kind_for_path
from core.adapters.pdf import VISION_PENDING_KEY
from core.adapters.stub import StubAdapter
from core.config import Settings, load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.models import Fragment, Material, MediaAsset, User
from core.db.session import session_scope
from core.ingest import handlers as h
from core.ingest.handlers import ASSETS_ORIGIN_KEY
from core.ingest.pipeline import StageHandler, process_material
from core.learning.checking import InvalidAnswerError
from core.learning.explain import format_duration
from core.learning.rating import Rating
from core.learning.scheduler import FixedIntervalScheduler
from core.logging import configure_logging, get_logger
from core.services.materials import create_or_find, mark_ready, merge_origin, sha256_of
from core.services.sessions import (
    AnswerResult,
    ItemView,
    SessionSummary,
    active_session_id,
    answer_item,
    assemble,
    current_item,
    mark_shown,
)
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

    kind = kind_for_path(str(path))
    adapter = build_adapter(kind, settings=settings)
    fragments: list[ExtractedFragment] = []
    if adapter is None:
        # Текстовый материал: §5 адаптера для него не вводит, разбирает сам
        # харнесс. Читать как текст **до** проверки типа нельзя — PDF падал бы
        # на `read_text` с `UnicodeDecodeError`, что и происходило до WP-06.
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
                kind=kind,
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

        if adapter is None:
            adapter = StubAdapter(
                meta=SourceMeta(text_chars=sum(len(item.text or "") for item in fragments)),
                fragments=fragments,
            )
        else:
            # Замер источника и запись его в `materials.origin` (§5.1): число
            # страниц и страницы, ждущие vision. Гейт бюджета §6.3 здесь
            # **не** вызывается сознательно: он задаёт вопрос человеку и ждёт
            # нажатия кнопки, а у консоли спрашивать некого — в воркере он
            # остаётся на месте (`core/ingest/gate.py`).
            meta = await adapter.probe(str(path))
            async with session_scope(sessions) as session:
                await merge_origin(session, material_id, meta.extra)
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

    print(f"готово: material_id={material_id}, тип={kind}")
    return material_id


async def dump_material(material_id: int) -> int:
    """Печатает фрагменты материала читаемо. План §1.4, критерий WP-06.

    Нужен ровно потому, что агент не видит ни Telegram, ни базы глазами: §1.4
    Плана называет консоль «основным способом разработки». Здесь — то, по чему
    судят о качестве разбора PDF: номер страницы, заголовок, качество, число
    токенов и начало текста.

    Страницы, ждущие vision, печатаются отдельной строкой: §6.3 требует, чтобы
    последствия были видимыми, а до PR B пакета такие страницы пропускаются.
    """
    settings = load_settings()
    engine = build_engine(settings.DATABASE_URL)
    sessions = build_sessionmaker(engine)

    try:
        async with session_scope(sessions) as session:
            row = (
                await session.execute(
                    select(Material.kind, Material.status, Material.origin).where(
                        Material.id == material_id
                    )
                )
            ).first()
            if row is None:
                print(f"материала id={material_id} нет")
                return 1
            kind, status, origin = row

            assets = (
                await session.execute(
                    select(MediaAsset.id, MediaAsset.path, MediaAsset.caption)
                    .where(MediaAsset.material_id == material_id)
                    .order_by(MediaAsset.id)
                )
            ).all()

            fragments = (
                await session.execute(
                    select(
                        Fragment.ord,
                        Fragment.kind,
                        Fragment.locator,
                        Fragment.quality,
                        Fragment.tokens,
                        Fragment.text,
                    )
                    .where(Fragment.material_id == material_id)
                    .order_by(Fragment.ord)
                )
            ).all()

        print(f"материал {material_id}: тип={kind}, статус={status}, фрагментов={len(fragments)}")
        pending = list((origin or {}).get(VISION_PENDING_KEY) or [])
        if pending:
            print(
                f"страниц без текстового слоя: {len(pending)} — "
                f"{', '.join(str(page) for page in pending)}; "
                "они ждут распознавания и заданий пока не дают"
            )
        if assets:
            items = ((origin or {}).get(ASSETS_ORIGIN_KEY) or {}).get("items") or []
            measurements = {
                int(item["asset_id"]): item for item in items if item.get("asset_id") is not None
            }
            print(_assets_block(assets, measurements, MATERIALS_ROOT))
        print()
        for ord_, frag_kind, locator, quality, tokens, text in fragments:
            print(_fragment_line(ord_, frag_kind, locator or {}, quality, tokens, text))
    finally:
        await engine.dispose()
    return 0


def _assets_block(
    rows: Sequence[tuple[int, str | None, str | None]],
    measurements: dict[int, dict[str, Any]],
    root: str,
) -> str:
    """Ассеты материала: страница, площадь, подпись, группа и наличие файла.

    Путь в базе хранится **относительно корня хранилища** (решение C1b), и
    склеивать его нужно здесь, а не в базе: воркер работает в контейнере,
    харнесс — на хосте. Пометка «файла нет» и есть обнаружение окна §4.2 —
    строки коммитятся до переноса файлов, и падение процесса между этими
    шагами оставляет ссылку без файла. §6.3 требует видимых последствий.

    Перечень измерений (страница, доля площади, группа) берётся из
    `materials.origin` и сопоставляется со строками по `asset_id`: колонок под
    них в `media_assets` нет, постоянное место группы — локатор фрагмента
    (§3.4), и это долг C2 (ADR-0008). Ассет, которого нет в перечне, печатается
    с «стр. —»: это видимое расхождение между таблицей и `origin`, а не
    молчаливый пропуск.

    Первая редакция этой функции измерения **не читала**, хотя докстрока
    утверждала обратное, — и приёмка «группа из десяти на стр. 10 видна в
    `dump-material`» не выполнялась. Нашло ревью PR #53; тесты были зелёными
    потому, что проверяли только число ассетов и пометку о файле.
    """
    lines = [f"ассетов: {len(rows)}"]
    for asset_id, path, caption in rows:
        exists = (Path(root) / path).is_file() if path else False
        mark = "" if exists else "  ⚠️ файла нет по пути"
        shown = (caption or "").strip().replace("\n", " ")
        tail = f" «{shown[:40]}»" if shown else ""

        item = measurements.get(asset_id, {})
        page = item.get("page")
        ratio = item.get("area_ratio")
        group = item.get("group_id")
        where = f"стр. {page}" if page is not None else "стр. —"
        area = f" {float(ratio):.4f}" if isinstance(ratio, (int, float)) else ""
        band = f" [{group}]" if group else ""

        lines.append(f"  id={asset_id} {where}{area}{band} {path}{tail}{mark}")
    return "\n".join(lines)


def _fragment_line(
    ord_: int,
    kind: str | None,
    locator: dict[str, object],
    quality: float | None,
    tokens: int | None,
    text: str | None,
) -> str:
    """Одна строка вывода `dump-material`.

    Форма зависит от варианта локатора §3.4: у `pdf` это страница, у `image` —
    ассет и область. Неизвестный вариант печатается как есть, а не
    пропускается: пропуск скрыл бы фрагмент, который существует.
    """
    where = {
        "pdf": lambda: f"стр. {locator.get('page')}",
        "image": lambda: f"ассет {locator.get('asset_id')}",
        "web": lambda: f"«{str(locator.get('quote') or '')[:40]}»",
    }.get(str(locator.get("type")), lambda: f"локатор {locator.get('type')}")()

    heading = locator.get("heading")
    head = f" [{heading}]" if heading else ""
    body = " ".join((text or "").split())[:80]
    return (
        f"{ord_:>3}. {kind or '?':<6} {where:<16}{head} "
        f"q={quality if quality is not None else '?'} t={tokens or 0}\n     {body}"
    )


async def show_session(tg_id: int | None = None) -> int:
    """Собирает или продолжает сеанс и печатает текущее задание.

    Возвращает код выхода: 0 — задание напечатано, 1 — повторять нечего. Код
    возврата, а не только текст: цель `make session` вызывается из скриптов, и
    «нет заданий» там обязано отличаться от «задание выдано».

    Повторный вызов при незавершённом сеансе печатает **то же** задание и
    счётчик показов не растит — §13.3 сортирует вопросы по `shown_count`, и
    перезагрузка экрана не должна влиять на порядок выдачи.
    """
    settings = load_settings()
    engine = build_engine(settings.DATABASE_URL)
    sessions = build_sessionmaker(engine)
    now = datetime.now(UTC)

    try:
        async with session_scope(sessions) as session:
            user_id = await _user_by_tg_id(session, _owner_tg_id(settings, tg_id))
            session_id = await assemble(session, user_id=user_id, settings=settings, now=now)
            if session_id is None:
                print(
                    "Повторять нечего: кандидатов меньше "
                    f"MIN_SESSION_ITEMS = {settings.MIN_SESSION_ITEMS} (§16)."
                )
                return 1

            item = await current_item(session, session_id)
            if item is None:
                print(f"Сеанс {session_id} уже доигран — заданий без ответа нет.")
                return 1

            await mark_shown(session, item.item_id, user_id=user_id, now=now)
            print(_render_item(session_id, item))
    finally:
        await engine.dispose()
    return 0


async def answer(*, option: int | None = None, idk: bool = False, tg_id: int | None = None) -> int:
    """Отвечает на текущее задание активного сеанса и печатает разбор.

    Ровно один из `option` и `idk`: «не знаю» — отдельный исход §13.2, а не
    вариант ответа, и путать их нельзя, иначе в журнале `verdict` перестанет
    отличать пропуск от ошибки.
    """
    if idk == (option is not None):
        raise SystemExit("укажите либо OPTION=<номер варианта>, либо IDK=1")

    settings = load_settings()
    engine = build_engine(settings.DATABASE_URL)
    sessions = build_sessionmaker(engine)
    now = datetime.now(UTC)
    # `option` непуст: взаимное исключение проверено выше, и mypy это не видит.
    assert option is not None or idk
    raw: dict[str, object] = {"idk": True} if idk else {"option": int(option or 0) - 1}

    try:
        async with session_scope(sessions) as session:
            user_id = await _user_by_tg_id(session, _owner_tg_id(settings, tg_id))
            session_id = await active_session_id(session, user_id)
            if session_id is None:
                print("Активного сеанса нет — соберите его командой `make session`.")
                return 1

            item = await current_item(session, session_id)
            if item is None:
                print(f"Сеанс {session_id} уже доигран — отвечать не на что.")
                return 1

            try:
                result = await answer_item(
                    session,
                    user_id=user_id,
                    item_id=item.item_id,
                    raw=raw,
                    now=now,
                    settings=settings,
                    scheduler=FixedIntervalScheduler(),
                )
            except InvalidAnswerError as error:
                # Ответ не разобран: испорченная строка вопроса или номер
                # варианта вне диапазона. Печатаем причину и уходим с кодом 1 —
                # трассировка здесь ничего не добавляет, а причину скрывает.
                print(f"Ответ не разобран: {error}")
                return 1
            if result is None:
                print("Ответ не принят: задание уже отвечено или сеанс не активен (§18).")
                return 1

            print(_render_answer(result))
            if not result.finished:
                nxt = await current_item(session, session_id)
                if nxt is not None:
                    await mark_shown(session, nxt.item_id, user_id=user_id, now=now)
                    print()
                    print(_render_item(session_id, nxt))
    finally:
        await engine.dispose()
    return 0


def _owner_tg_id(settings: Settings, tg_id: int | None) -> int:
    """Telegram id владельца: переданный или первый из белого списка (§1.3)."""
    return tg_id if tg_id is not None else settings.ALLOWED_USER_IDS[0]


async def _user_by_tg_id(session: object, tg_id: int) -> int:
    """Существующий пользователь. Нового здесь не создаём.

    В отличие от `seed`, где материал появляется вместе с владельцем, сеанс
    повторения по несуществующему пользователю смысла не имеет: расписания у
    него нет. Создать его молча значило бы печатать «повторять нечего» вместо
    «такого пользователя нет».
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    assert isinstance(session, AsyncSession)
    found = (await session.execute(select(User.id).where(User.tg_id == tg_id))).scalar_one_or_none()
    if found is None:
        raise SystemExit(f"пользователя с tg_id={tg_id} нет — сначала `make seed`")
    return int(found)


def _render_item(session_id: int, item: ItemView) -> str:
    """Задание в консоли. Разметка §20 для mcq, без клавиатуры."""
    lines = [
        f"Сеанс {session_id} — задание {item.position} из {item.total}",
        "",
        str(item.payload.get("question", "")),
        "",
    ]
    options = list(item.payload.get("options") or [])
    lines.extend(f"  {number}. {text}" for number, text in enumerate(options, start=1))
    lines += [
        "",
        f"Ответ: make answer OPTION=1..{len(options)}  или  make answer IDK=1",
    ]
    return "\n".join(lines)


VERDICT_LABEL = {
    "correct": "✅ Верно",
    "wrong": "❌ Неверно",
    "skipped": "🤷 Не знаю",
    "partial": "◐ Частично",
}
"""Подписи вердиктов §22. `partial` появится с открытым ответом в WP-15 —
подпись заведена здесь, чтобы разбор не падал на незнакомом вердикте."""


def _render_answer(result: AnswerResult) -> str:
    """Разбор §22 и, если сеанс кончился, итог §22.1."""
    lines = [
        VERDICT_LABEL.get(result.verdict, result.verdict),
        f"Правильный ответ: {result.explanation.correct_answer}",
        "",
        f"Почему: {result.explanation.detail}",
    ]
    lines.extend(f"В источнике: {hint}" for hint in result.explanation.locator_hints)
    lines += ["", f"Рейтинг: {Rating(result.rating).name}"]

    if result.summary is not None:
        lines += ["", _render_summary(result.summary)]
    return "\n".join(lines)


def _render_summary(summary: SessionSummary) -> str:
    """Итог сеанса §22.1: счёт, время, список ошибок."""
    lines = [
        "— Сеанс завершён —",
        f"Счёт: {summary.correct} из {summary.total}",
        f"Время: {format_duration(summary.seconds)}",
    ]
    if summary.mistakes:
        lines.append("Ошибки:")
        lines.extend(f"  — {statement}" for statement in summary.mistakes)
    return "\n".join(lines)


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

    session_parser = sub.add_parser("session", help="собрать сеанс и показать задание")
    session_parser.add_argument("--tg-id", type=int, default=None, help="Telegram id")

    dump_parser = sub.add_parser("dump-material", help="показать фрагменты материала")
    dump_parser.add_argument("material_id", type=int, help="идентификатор материала")

    answer_parser = sub.add_parser("answer", help="ответить на текущее задание")
    answer_parser.add_argument("--option", type=int, default=None, help="номер варианта, с 1")
    answer_parser.add_argument("--idk", action="store_true", help="«не знаю» (§20)")
    answer_parser.add_argument("--tg-id", type=int, default=None, help="Telegram id")

    args = parser.parse_args()
    if args.command == "seed":
        asyncio.run(seed(args.file, args.tg_id))
    elif args.command == "dump-material":
        raise SystemExit(asyncio.run(dump_material(args.material_id)))
    elif args.command == "session":
        raise SystemExit(asyncio.run(show_session(args.tg_id)))
    elif args.command == "answer":
        raise SystemExit(asyncio.run(answer(option=args.option, idk=args.idk, tg_id=args.tg_id)))


if __name__ == "__main__":
    main()
