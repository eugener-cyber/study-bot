"""Ожидание ответа пользователя посреди конвейера. ТЗ §4.5.

Две ситуации требуют ответа: подтверждение бюджета (§6.3) и выбор языка при
неуверенном определении (§5.4). Обе используют **один** механизм — §4.5 так и
написано, и это существенно: два похожих механизма ожидания разошлись бы в
поведении при тайм-ауте, а тайм-аут здесь сутки.

Устройство: материал остаётся в статусе `processing` с
`progress.stage = 'awaiting_user'` и `progress.question`, задача ARQ
**завершается**, а продолжение ставит в очередь обработчик нажатия кнопки.

Фоновый воркер не блокируется в ожидании — это прямое требование §4.5 и
единственный способ не держать слот воркера сутки. Блокирующее ожидание
выглядело бы проще в коде и означало бы, что один материал на подтверждении
занимает воркер целиком.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings
from core.db.models import Material
from core.logging import get_logger

log = get_logger(__name__)

AWAITING_STAGE = "awaiting_user"
"""Значение `progress.stage` при ожидании. §4.5 называет его буквально."""


async def park(
    session: AsyncSession,
    material_id: int,
    question: str,
    *,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Ставит материал на ожидание ответа.

    `kind` различает, чего именно ждут: `budget` (§6.3) или `language` (§5.4).
    Без него обработчик нажатия не знал бы, что делать с ответом, и пришлось
    бы угадывать по тексту вопроса — то есть по строке, предназначенной
    человеку.

    Статус материала остаётся `processing`. Не `failed` и не отдельный статус:
    §4.1 перечисляет четыре статуса, и пятый сломал бы все выборки по
    `status`, написанные по спецификации.
    """
    progress: dict[str, Any] = {
        "stage": AWAITING_STAGE,
        "question": question,
        "kind": kind,
        "since": datetime.now(UTC).isoformat(),
    }
    if payload:
        progress["payload"] = payload

    await session.execute(
        update(Material).where(Material.id == material_id).values(progress=progress)
    )
    log.info("material_awaiting_user", material_id=material_id, kind=kind)


async def resume(session: AsyncSession, material_id: int) -> None:
    """Снимает ожидание. Вызывается обработчиком нажатия кнопки.

    `progress` затирается целиком, а не правится поле `stage`: остатки
    `question` и `payload` от прошлого ожидания попали бы в следующее, и
    пользователь увидел бы вопрос о бюджете при выборе языка.
    """
    await session.execute(update(Material).where(Material.id == material_id).values(progress={}))
    log.info("material_resumed", material_id=material_id)


async def _progress(session: AsyncSession, material_id: int) -> dict[str, Any]:
    """`materials.progress` материала, всегда словарём.

    `None` приводится к пустому словарю: §3 объявляет поле nullable, и каждому
    вызывающему иначе пришлось бы проверять это самому — а забытая проверка
    даёт `AttributeError` на `None.get`.
    """
    result = await session.execute(select(Material.progress).where(Material.id == material_id))
    raw: dict[str, Any] | None = result.scalar_one()
    return dict(raw) if raw else {}


async def is_awaiting(session: AsyncSession, material_id: int) -> bool:
    return (await _progress(session, material_id)).get("stage") == AWAITING_STAGE


async def awaiting_kind(session: AsyncSession, material_id: int) -> str | None:
    """Чего именно ждут от пользователя, или `None`, если не ждут."""
    progress = await _progress(session, material_id)
    if progress.get("stage") != AWAITING_STAGE:
        return None
    kind = progress.get("kind")
    return str(kind) if kind else None


async def expire_stale(session: AsyncSession, settings: Settings) -> list[int]:
    """Переводит в `failed` материалы, ждущие дольше `AWAITING_USER_TIMEOUT_H`.

    §4.5: «Если ответа нет `AWAITING_USER_TIMEOUT_H` часов, материал
    переводится в `failed` с понятной причиной; повторная обработка доступна
    кнопкой».

    «Понятная причина» — не формальность: материал в `failed` без текста
    выглядит сбоем системы, хотя сбоя не было, и пользователь будет искать
    поломку вместо того, чтобы нажать кнопку.

    Сравнение по `progress->>'since'`, а не по `created_at`: ждать начали не
    при создании материала, а при постановке вопроса, и между этими моментами
    лежит вся стадия `probe` плюс очередь.
    """
    deadline = datetime.now(UTC) - timedelta(hours=settings.AWAITING_USER_TIMEOUT_H)

    rows = (
        await session.execute(
            select(Material.id, Material.progress).where(
                Material.status == "processing",
                Material.progress["stage"].astext == AWAITING_STAGE,
            )
        )
    ).all()

    expired: list[int] = []
    for material_id, progress in rows:
        since = (progress or {}).get("since")
        if not since:
            continue
        if datetime.fromisoformat(str(since)) < deadline:
            expired.append(int(material_id))

    if not expired:
        return []

    await session.execute(
        update(Material)
        .where(Material.id.in_(expired))
        .values(
            status="failed",
            error=(
                "Обработка остановлена: не получен ответ на вопрос о материале "
                f"в течение {settings.AWAITING_USER_TIMEOUT_H} часов. "
                "Нажмите «Обработать заново», чтобы начать снова."
            ),
            progress={},
        )
    )
    log.info("awaiting_expired", count=len(expired), material_ids=expired)
    return expired
