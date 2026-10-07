"""Материалы: создание, дубликаты, перенос между предметами. ТЗ §4.4.

§35: доступ к БД только через сервисный слой. Хендлер разбирает апдейт,
вызывает отсюда и рендерит ответ.

Дубликат определяется по `sha256` **в рамках пользователя** — §4.4. Не
глобально: один и тот же файл у двух людей это два материала с отдельной
историей обучения. Индекс под это есть с WP-02 и частичный по наличию ключа,
потому что у материала без файла (ссылки) ограничения быть не должно.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import Material
from core.logging import get_logger

log = get_logger(__name__)

CHUNK = 1024 * 1024
"""Размер блока при чтении файла.

Мегабайт, а не весь файл сразу: §2.2 поднимает лимит до 2 ГБ, и чтение
лекционного видео целиком в память положило бы процесс. Это не оптимизация,
а условие работоспособности на заявленном лимите.
"""


def sha256_of(path: str | Path) -> str:
    """Хеш файла блоками."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


async def find_duplicate(session: AsyncSession, user_id: int, sha256: str) -> Material | None:
    """Существующий материал пользователя с тем же `sha256`, или `None`."""
    return (
        await session.execute(
            select(Material).where(
                Material.user_id == user_id,
                Material.origin["sha256"].astext == sha256,
            )
        )
    ).scalar_one_or_none()


async def create_or_find(
    session: AsyncSession,
    *,
    user_id: int,
    kind: str,
    title: str,
    origin: dict[str, Any],
    subject_id: int | None = None,
) -> tuple[Material, bool]:
    """Создаёт материал или возвращает существующий дубликат.

    Второй элемент — признак дубликата: `True`, если материал уже был.
    Признак возвращается, а не выводится вызывающим по косвенным признакам:
    §4.4 требует показать существующий материал с кнопками, и хендлер должен
    знать, какой из двух экранов рисовать.

    Копия не создаётся. §4.4: «Показывается существующий материал с кнопками
    `[Открыть]` и `[Перенести в другой предмет]`; второй вариант меняет только
    `subject_id`, копия не создаётся».

    Материал **без** `sha256` в `origin` — ссылка или текст — дубликатом
    никогда не считается: уникальный индекс §3.1 частичный по наличию ключа,
    и две ссылки на одну страницу это два материала. Проверять их на
    совпадение по URL спецификация не требует, и выдумывать это правило
    здесь нельзя.
    """
    sha256 = origin.get("sha256")
    if sha256:
        existing = await find_duplicate(session, user_id, str(sha256))
        if existing is not None:
            log.info("material_duplicate", material_id=existing.id, user_id=user_id, kind=kind)
            return existing, True

    material = Material(
        user_id=user_id,
        subject_id=subject_id,
        kind=kind,
        title=title,
        origin=origin,
        status="queued",
        mode="full",
        completed_stages=[],
        progress={},
    )
    session.add(material)
    await session.flush()
    log.info("material_created", material_id=material.id, user_id=user_id, kind=kind)
    return material, False


async def move_to_subject(session: AsyncSession, material_id: int, subject_id: int | None) -> None:
    """Меняет предмет материала. §4.4, кнопка «Перенести в другой предмет».

    Меняется ровно `subject_id`. Ни `status`, ни `completed_stages`, ни
    артефакты: перенос — это смена папки, а не переобработка. Обратное
    означало бы потерю истории обучения при нажатии кнопки, которая выглядит
    безобидной.
    """
    await session.execute(
        update(Material).where(Material.id == material_id).values(subject_id=subject_id)
    )
    log.info("material_moved", material_id=material_id, subject_id=subject_id)


async def mark_ready(session: AsyncSession, material_id: int) -> None:
    """Переводит материал в `ready` и проставляет `processed_at` (§4.1).

    `processed_at` ставится здесь, а не на стадии `schedule`: §4.6 считает от
    него `due_at`, но сама стадия идёт раньше перехода в `ready`, и на её
    момент поле ещё пусто. Расхождение в доли секунды зафиксировано в
    докстроке `ensure_review_states`.
    """
    from sqlalchemy import func

    await session.execute(
        update(Material)
        .where(Material.id == material_id)
        .values(status="ready", processed_at=func.now(), progress={}, error=None)
    )
    log.info("material_ready", material_id=material_id)


async def mark_failed(session: AsyncSession, material_id: int, reason: str) -> None:
    """Переводит материал в `failed` с причиной (§4.1).

    Причина обязательна параметром, а не необязательна: материал в `failed`
    без текста выглядит сбоем системы, и пользователь будет искать поломку
    вместо того, чтобы нажать «Обработать заново».
    """
    await session.execute(
        update(Material)
        .where(Material.id == material_id)
        .values(status="failed", error=reason, progress={})
    )
    log.info("material_failed", material_id=material_id)
