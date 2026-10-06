"""Фоновый воркер. ТЗ §1.4, план реализации §1.4.

Второй процесс того же образа — §1.4 ТЗ: «четыре контейнера: `postgres`,
`redis`, `telegram-bot-api`, `app` (бот и ARQ-воркер как два процесса одного
образа)».

ARQ, а не Celery: §1.4 ТЗ отвергает Celery прямо — «ARQ работает в том же
asyncio-цикле без второго рантайма».

Воркер **не блокируется** в ожидании ответа пользователя (§4.5). Когда стадия
требует подтверждения, задача завершается, а продолжение ставит в очередь
обработчик нажатия кнопки. Это не оптимизация: тайм-аут ожидания сутки
(`AWAITING_USER_TIMEOUT_H`), и блокирующее ожидание держало бы слот воркера
всё это время.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from arq import cron
from arq.connections import RedisSettings

from core.adapters.stub import StubAdapter
from core.config import Settings, load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.session import session_scope
from core.harness import fragments_from_text
from core.ingest import handlers as h
from core.ingest.awaiting import expire_stale
from core.ingest.pipeline import AwaitingUser, StageHandler, process_material
from core.logging import configure_logging, get_logger
from core.services.materials import mark_failed, mark_ready
from core.storage import MaterialStorage

log = get_logger(__name__)

MATERIALS_ROOT = "./data/materials"
QUEUE_NAME = "studybot:ingest"


def stage_handlers(settings: Settings) -> dict[str, StageHandler]:
    """Обработчики стадий. Содержательные — заглушки до WP-04."""
    return {
        "extract": h.extract,
        "sections": h.sections_stub,
        "facts": h.facts_stub,
        "notes": h.notes_stub,
        "questions": h.questions_stub,
        "schedule": h.make_schedule_handler(settings),
    }


async def process_material_task(ctx: dict[str, Any], material_id: int, source_path: str) -> str:
    """Задача обработки материала.

    Возвращает строку состояния, а не `None`: ARQ пишет результат в Redis, и
    по нему видно, чем кончилась задача, без чтения логов.

    `AwaitingUser` не ошибка и в `failed` материал не переводит: §4.5 требует
    оставить его в `processing` и ждать ответа. Логируется отдельным событием,
    чтобы в логе было видно разницу между «ждём человека» и «сломалось».
    """
    settings: Settings = ctx["settings"]
    sessions = ctx["sessions"]
    storage: MaterialStorage = ctx["storage"]

    adapter = StubAdapter(fragments=fragments_from_text_or_stub(source_path))

    try:
        await process_material(
            sessions,
            storage,
            stage_handlers(settings),
            material_id,
            adapter,
            source_path,
        )
    except AwaitingUser as awaiting:
        log.info("task_parked", material_id=material_id, question=awaiting.question)
        return "awaiting_user"
    except Exception as error:
        log.error("task_failed", material_id=material_id, error_type=type(error).__name__)
        async with session_scope(sessions) as session:
            await mark_failed(
                session,
                material_id,
                "Не удалось обработать материал. Нажмите «Обработать заново».",
            )
        raise

    async with session_scope(sessions) as session:
        await mark_ready(session, material_id)
    return "ready"


def fragments_from_text_or_stub(source_path: str) -> list[Any]:
    """Фрагменты из файла, если он текстовый, иначе заглушка.

    До WP-06 настоящего PDF-адаптера нет. Текстовые файлы при этом читаются
    по-настоящему — это позволяет прогонять конвейер на конспектах,
    подготовленных вручную для провайдера `manual` (§2.1, CR-H), а не только
    на выдуманном фрагменте.
    """
    from pathlib import Path

    path = Path(source_path)
    if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
        return fragments_from_text(path)
    return []


async def expire_awaiting_task(ctx: dict[str, Any]) -> int:
    """Переводит в `failed` материалы, ждущие ответа дольше тайм-аута (§4.5).

    Отдельная периодическая задача, а не проверка при каждом обращении: §4.5
    задаёт тайм-аут в часах, и материал может ждать, пока пользователь не
    открывает бота вовсе — то есть пока никаких обращений нет.
    """
    settings: Settings = ctx["settings"]
    async with session_scope(ctx["sessions"]) as session:
        expired = await expire_stale(session, settings)
    return len(expired)


async def startup(ctx: dict[str, Any]) -> None:
    configure_logging()
    settings = load_settings()
    engine = build_engine(settings.DATABASE_URL)

    ctx["settings"] = settings
    ctx["engine"] = engine
    ctx["sessions"] = build_sessionmaker(engine)
    ctx["storage"] = MaterialStorage(MATERIALS_ROOT)
    log.info("worker_started", queue=QUEUE_NAME)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["engine"].dispose()
    log.info("worker_stopped")


DEV_REDIS_URL = "redis://127.0.0.1:6379/0"
"""Запасной адрес Redis на случай отсутствия переменной.

`REDIS_URL` обязателен по §30.2, и в работе он всегда есть. Но ARQ читает
`redis_settings` **в момент импорта** класса настроек, а полная валидация
`Settings` на импорте сломала бы импорт модуля в тестах, где окружение
подставляется фикстурой позже. Поэтому здесь читается одна переменная
напрямую, а не весь конфиг.
"""


class WorkerSettings:
    """Настройки ARQ. Класс, а не словарь — так требует сам ARQ."""

    functions: ClassVar[list[Any]] = [process_material_task]
    cron_jobs: ClassVar[list[Any]] = [
        # §4.5: материал может ждать ответа, пока пользователь не открывает
        # бота вовсе. Поэтому истечение проверяется по расписанию, а не при
        # обращении — иначе материал ждал бы вечно именно в том случае, для
        # которого тайм-аут и вводился.
        cron(expire_awaiting_task, hour=None, minute={0, 30}, run_at_startup=True)
    ]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", DEV_REDIS_URL))
