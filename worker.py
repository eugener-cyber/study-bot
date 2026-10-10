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

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.handlers.gate import gate_keyboard
from bot.main import LOCAL_BOT_API
from bot.progress import stage_text
from core.adapters.stub import StubAdapter
from core.config import Settings, load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.models import Material, User
from core.db.session import session_scope
from core.harness import fragments_from_text
from core.ingest import handlers as h
from core.ingest.awaiting import expire_stale, progress_message_id
from core.ingest.gate import probe_and_gate, reestimate_after_extract
from core.ingest.pipeline import AwaitingUser, StageHandler, process_material
from core.ingest.stages import EXTRACT
from core.llm.factory import build_client
from core.logging import configure_logging, get_logger
from core.services.materials import mark_failed, mark_ready
from core.storage import MaterialStorage

log = get_logger(__name__)

MATERIALS_ROOT = "./data/materials"
QUEUE_NAME = "studybot:ingest"


def stage_handlers(
    settings: Settings, sessions: async_sessionmaker[AsyncSession]
) -> dict[str, StageHandler]:
    """Обработчики стадий §4.2 с клиентом LLM.

    `sessions` нужны не стадиям, а учёту: `DatabaseRecorder` пишет строку
    `llm_calls` в своей транзакции, чтобы падение стадии не отменяло откатом
    запись об уже оплаченных вызовах (урок WP-03 с `AwaitingUser`).
    """
    return h.build_handlers(build_client(settings, sessions), settings)


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
        # Гейт бюджета — ДО конвейера, а не внутри. §6.3: для сканированного
        # PDF `extract` это vision-вызов на каждую страницу, и гейт после него
        # спрашивал бы «обрабатывать?» после того, как деньги потрачены.
        # Карта стадий строится один раз на задачу. Два вызова `stage_handlers`
        # дали бы два клиента, то есть два независимых семафора
        # `LLM_CONCURRENCY`, и предел параллелизма §6.1 оказался бы вдвое
        # выше заявленного — молча, потому что каждый из них соблюдает свой.
        handlers = stage_handlers(settings, sessions)

        preliminary = await probe_and_gate(sessions, material_id, adapter, source_path, settings)

        # Извлечение отдельно от остальных стадий: сразу после него оценка
        # пересчитывается по фактическим фрагментам, и при расхождении больше
        # `COST_REESTIMATE_FACTOR` пользователь спрашивается повторно. Без
        # этого гейт обходится любым материалом, чью стоимость `probe`
        # недооценил (§6.3).
        await process_material(
            sessions,
            storage,
            handlers,
            material_id,
            adapter,
            source_path,
            stages=(EXTRACT,),
        )
        await reestimate_after_extract(sessions, material_id, preliminary, settings)
        await _show_stage(ctx, material_id, "sections")

        await process_material(
            sessions,
            storage,
            handlers,
            material_id,
            adapter,
            source_path,
        )
    except AwaitingUser as awaiting:
        # Вопрос обязан дойти до пользователя: без него материал стоит в
        # `awaiting_user` молча, а §4.5 на этом и держится — продолжение
        # ставит в очередь обработчик нажатия кнопки, и без кнопки его не
        # будет. Тайм-аут тогда переведёт материал в `failed` через сутки, и
        # выглядеть это будет сбоем системы.
        await _ask_user(ctx, material_id, awaiting.question)
        log.info("task_parked", material_id=material_id)
        return "awaiting_user"
    except Exception as error:
        log.error("task_failed", material_id=material_id, error_type=type(error).__name__)
        # `mark_failed` в СВОЕЙ транзакции: сбой случился в другой, и та уже
        # откачена. Запись в той же границе была бы откачена вместе со сбоем,
        # и материал остался бы в `processing` навсегда — §4.1 такого
        # состояния не предусматривает.
        async with session_scope(sessions) as session:
            await mark_failed(
                session,
                material_id,
                "Не удалось обработать материал. Нажмите «Обработать заново».",
            )
        await _show_failed(ctx, material_id)
        raise

    async with session_scope(sessions) as session:
        await mark_ready(session, material_id)
    await _show_done(ctx, material_id)
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


async def _edit_progress(ctx: dict[str, Any], material_id: int, text: str) -> bool:
    """Правит сообщение прогресса материала. Возвращает, получилось ли.

    Воркер правит **то же** сообщение, которое создал хендлер (§4.5: «одно
    сообщение»). Идентификатор берётся из `progress.message_id` — его положил
    туда хендлер при постановке задачи.

    `None` вместо идентификатора — нормальный исход: материал мог прийти через
    `make seed`, где Telegram не участвует вовсе. Сбой правки гасится: §4.5
    про отображение прогресса, и падать из-за него посреди обработки значило
    бы терять материал из-за косметики.
    """
    bot: Bot | None = ctx.get("bot")
    if bot is None:
        return False

    async with session_scope(ctx["sessions"]) as session:
        message_id = await progress_message_id(session, material_id)
        chat_id = (
            await session.execute(
                select(User.tg_id)
                .join(Material, Material.user_id == User.id)
                .where(Material.id == material_id)
            )
        ).scalar_one_or_none()

    if message_id is None or chat_id is None:
        return False

    try:
        await bot.edit_message_text(text=text, chat_id=int(chat_id), message_id=message_id)
    except Exception:
        log.info("progress_not_edited", material_id=material_id)
        return False
    return True


async def _show_stage(ctx: dict[str, Any], material_id: int, stage: str) -> None:
    await _edit_progress(ctx, material_id, stage_text(stage))


async def _show_done(ctx: dict[str, Any], material_id: int) -> None:
    await _edit_progress(ctx, material_id, "✅ Готово. Материал обработан, задания в расписании.")


async def _show_failed(ctx: dict[str, Any], material_id: int) -> None:
    await _edit_progress(
        ctx,
        material_id,
        "Не удалось обработать материал. Нажмите «Обработать заново».",
    )


async def _ask_user(ctx: dict[str, Any], material_id: int, question: str) -> None:
    """Задаёт пользователю вопрос гейта с кнопками §6.3.

    Сбой отправки гасится и логируется: исключение здесь заменило бы
    «ждём ответа» на «задача упала», и материал ушёл бы в `failed` при живом
    гейте. Пользователь в этом случае увидит вопрос позже — после тайм-аута
    и повторной обработки, — но данные не потеряются.
    """
    bot: Bot | None = ctx.get("bot")
    if bot is None:
        log.info("question_not_sent_no_bot", material_id=material_id)
        return

    async with session_scope(ctx["sessions"]) as session:
        tg_id = (
            await session.execute(
                select(User.tg_id)
                .join(Material, Material.user_id == User.id)
                .where(Material.id == material_id)
            )
        ).scalar_one_or_none()

    if tg_id is None:
        log.info("question_not_sent_no_owner", material_id=material_id)
        return

    try:
        await bot.send_message(
            chat_id=int(tg_id),
            text=f"{question}\n\nЧто делать?",
            reply_markup=gate_keyboard(material_id),
        )
    except Exception:
        log.error("question_not_sent", material_id=material_id, exc_info=True)


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

    # Воркеру нужен бот: гейт бюджета задаёт вопрос, и задавать его больше
    # некому — хендлер к этому моменту давно ответил пользователю и вышел.
    # Сессия та же, что у бота: локальный Bot API (§2.2).
    ctx["bot"] = Bot(
        token=settings.BOT_TOKEN,
        session=AiohttpSession(api=TelegramAPIServer.from_base(LOCAL_BOT_API)),
    )
    log.info("worker_started", queue=QUEUE_NAME)


async def shutdown(ctx: dict[str, Any]) -> None:
    bot: Bot | None = ctx.get("bot")
    if bot is not None:
        await bot.session.close()
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
