"""Приём материала. ТЗ §2.2, §4.4, §4.5, §26.

§26: «Загрузка работает без команды: присланный файл, фото, голосовое или
ссылка → выбор предмета → обработка». Поэтому здесь нет команды — только
фильтры по содержимому сообщения.

§2.2: файл скачивается через **локальный** Bot API server. Это поднимает лимит
с 20 МБ до 2 ГБ и отдаёт путь на диске вместо HTTP-скачивания. Обе части
существенны: без первой лекционное видео не принять вовсе, без второй файл
пришлось бы качать через сеть у сервера, который держит его на том же диске.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aiogram import F, Router
from aiogram.types import Message
from arq import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from bot.progress import stage_text
from core.logging import get_logger
from core.services.materials import create_or_find, sha256_of
from core.services.users import find_by_tg_id

log = get_logger(__name__)

INCOMING_ROOT = Path("./data/materials/incoming")
"""Куда кладётся присланный файл до создания материала.

Внутри тома §30.1, а не в системном `/tmp`: файл должен пережить перезапуск
контейнера между приёмом и обработкой. `/tmp` в контейнере этого не
гарантирует.
"""

NO_CONSENT = "Сначала нужно согласие на обработку данных — отправьте /start."
"""§31: согласие фиксируется при первом запуске.

Материал нельзя принять раньше согласия: приём означает хранение файла, то
есть начало обработки персональных данных, а согласие — ровно то, что это
разрешает.
"""

UNSUPPORTED_KIND = (
    "Пока я принимаю только документы и фотографии.\n\n"
    "Аудио, видео и ссылки — в следующих версиях."
)
"""Честный отказ вместо тишины.

Это то, чем переходный §19.4 был заменён: там бот молчал, потому что не умел
принимать ничего. Теперь он умеет принимать часть и говорит, какую именно.
"""


def _kind_of(message: Message) -> str | None:
    """Тип источника по §3: `pdf|image|audio|video|web|text`.

    `None` означает «не поддерживается здесь» — не ошибку. Аудио и видео
    появятся в WP-19 и WP-20, ссылки в WP-22, и до тех пор ответ о них должен
    быть внятным.
    """
    if message.document:
        name = (message.document.file_name or "").lower()
        if name.endswith(".pdf"):
            return "pdf"
        if name.endswith((".md", ".txt")):
            return "text"
        return None
    if message.photo:
        return "image"
    return None


async def _download(message: Message, file_id: str, suffix: str) -> Path:
    """Скачивает файл через локальный Bot API и возвращает путь.

    При `TELEGRAM_LOCAL=1` сервер уже держит файл на диске и отдаёт путь, а не
    ссылку (§2.2, ADR-0002). Поэтому здесь копирование с одного пути на
    другой, а не загрузка по сети: качать через HTTP у сервера, который лежит
    на том же диске, — лишний проход по гигабайтам.
    """
    assert message.bot is not None
    telegram_file = await message.bot.get_file(file_id)
    INCOMING_ROOT.mkdir(parents=True, exist_ok=True)
    target = INCOMING_ROOT / f"{file_id}{suffix}"

    source = telegram_file.file_path
    if source and Path(source).is_file():
        shutil.copyfile(source, target)
        return target

    # Публичный Bot API или файл вне локального пути: остаётся скачать.
    # Ветка существует, чтобы бот работал и без локального сервера — но §2.2
    # называет локальный обязательным, и лимит 20 МБ здесь вернётся.
    log.info("file_downloaded_over_http", file_id=file_id)
    await message.bot.download_file(str(source), destination=target)
    return target


async def handle_material(
    message: Message, session: AsyncSession, arq: ArqRedis | None = None
) -> None:
    """Принимает файл, создаёт материал и ставит обработку в очередь.

    Порядок намеренный: сначала проверка согласия, потом скачивание. Обратный
    означал бы, что файл человека без согласия уже лежит на диске, — то есть
    обработка персональных данных началась до разрешения.
    """
    user_id = message.from_user.id if message.from_user else None
    if user_id is None:
        return

    user = await find_by_tg_id(session, user_id)
    if user is None or user.consent_at is None:
        await message.answer(NO_CONSENT)
        return

    kind = _kind_of(message)
    if kind is None:
        await message.answer(UNSUPPORTED_KIND)
        log.info("unsupported_material", user_id=user_id)
        return

    if message.document:
        file_id = message.document.file_id
        name = message.document.file_name or file_id
        suffix = Path(name).suffix
    else:
        assert message.photo is not None
        file_id = message.photo[-1].file_id
        name = f"photo-{file_id[:8]}.jpg"
        suffix = ".jpg"

    progress = await message.answer(stage_text("probe"))
    path = await _download(message, file_id, suffix)

    material, duplicate = await create_or_find(
        session,
        user_id=user.id,
        kind=kind,
        title=name,
        origin={"sha256": sha256_of(path), "orig_name": name, "path": str(path)},
    )

    if duplicate:
        # §4.4: показывается существующий материал, копия не создаётся.
        await progress.edit_text(
            f"Этот файл уже есть: «{material.title}».\n\n"
            "Обрабатывать заново не нужно — он в вашем списке материалов."
        )
        return

    if arq is not None:
        await arq.enqueue_job("process_material_task", material.id, str(path))
        await progress.edit_text(stage_text("extract"))
        log.info("material_enqueued", material_id=material.id, kind=kind)
    else:
        # Воркера нет — материал остаётся в `queued`. Это честнее, чем
        # обрабатывать в хендлере: §4.5 требует, чтобы конвейер шёл фоном, и
        # обработка здесь блокировала бы ответы пользователю на минуты.
        await progress.edit_text(
            "Материал принят и ждёт обработки. Она начнётся, когда запустится воркер."
        )
        log.info("material_queued_without_worker", material_id=material.id)


def build_upload_router(arq: ArqRedis | None = None) -> Router:
    """Роутер приёма материалов.

    Фабрика — конвенция проекта с WP-01: зависимости передаются явно, а
    роутер уровня модуля делает сборку диспетчера непроверяемой.
    """
    router = Router(name="upload")

    async def on_material(message: Message, session: AsyncSession) -> None:
        await handle_material(message, session, arq)

    router.message.register(on_material, F.document | F.photo)
    return router
