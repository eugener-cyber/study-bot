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
from typing import Any, ClassVar, Final

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramAPIError
from aiogram.types import InlineKeyboardMarkup
from arq import cron
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.callbacks import retry_keyboard
from bot.handlers.gate import gate_keyboard
from bot.main import LOCAL_BOT_API
from bot.progress import stage_text
from core.adapters.base import SourceAdapter
from core.adapters.factory import (
    AUDIO_KIND,
    IMAGE_KIND,
    VIDEO_KIND,
    AdapterNotReadyError,
    build_adapter,
)
from core.adapters.pdf import VISION_PENDING_KEY
from core.adapters.stub import StubAdapter
from core.config import Settings, load_settings
from core.db.engine import build_engine, build_sessionmaker
from core.db.models import Material, User
from core.db.session import session_scope
from core.harness import fragments_from_text
from core.ingest import handlers as h
from core.ingest.awaiting import expire_stale, progress_message_id
from core.ingest.gate import probe_and_gate, reestimate_after_extract
from core.ingest.handlers import ASSETS_ORIGIN_KEY
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

    try:
        # Выбор адаптера — внутри `try`, а не перед ним: по §3 тип материала
        # лежит в `materials.kind`, и у типа без адаптера фабрика отказывает
        # (изображения до PR C, аудио и видео до WP-19 и WP-20). Отказ перед
        # `try` оставлял бы материал в `processing` навсегда — состояния
        # «обрабатывается, но никто не обрабатывает» §4.1 не предусматривает.
        async with session_scope(sessions) as session:
            kind = str(
                (
                    await session.execute(select(Material.kind).where(Material.id == material_id))
                ).scalar_one()
            )
        adapter = _adapter_for(kind, source_path, settings)

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
    except AdapterNotReadyError as error:
        # Тип материала известен §3, но адаптера для него ещё нет. Кнопка
        # «Обработать заново» здесь бессмысленна: повтор приведёт к тому же
        # отказу, пока адаптер не приедет. Поэтому причина называется прямо, а
        # кнопки нет (Issue #44). Различается **типом** исключения, а не
        # разбором текста — разбор был бы п. 7 списка костылей.
        log.info("task_unsupported_kind", material_id=material_id, kind=kind, error=str(error))
        async with session_scope(sessions) as session:
            await mark_failed(session, material_id, unsupported_kind_text(kind))
        await _show_failed(ctx, material_id, with_retry=False)
        raise
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


def _adapter_for(kind: str, source_path: str, settings: Settings) -> SourceAdapter:
    """Адаптер по `materials.kind`. §5.1, одна точка выбора — фабрика.

    Тип берётся из базы, а не выводится из пути заново: по §3 тип материала —
    это `materials.kind`, он записан при приёме, и второй вывод по имени файла
    расходился с ним на фотографиях (замечание ревью PR #42). Путь к файлу
    нужен здесь только текстовой заглушке.

    Заглушка остаётся для текстовых материалов: §5 адаптера для них не
    предусматривает, а конспекты, подготовленные вручную для провайдера
    `manual` (§2.1, CR-H), прогоняются именно так.
    """
    built = build_adapter(kind, settings=settings)
    if built is not None:
        return built
    return StubAdapter(fragments=fragments_from_text_or_stub(source_path))


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


async def _edit_progress(
    ctx: dict[str, Any],
    material_id: int,
    text: str,
    *,
    keyboard: InlineKeyboardMarkup | None = None,
) -> bool:
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
        await bot.edit_message_text(
            text=text,
            chat_id=int(chat_id),
            message_id=message_id,
            reply_markup=keyboard,
        )
    except TelegramAPIError as error:
        # Гасится **только** ошибка API: §4.5 про отображение прогресса, и
        # падать из-за косметики посреди обработки значило бы терять материал.
        # Широкий `except Exception` был п. 4 списка костылей и успел это
        # доказать: при добавлении `reply_markup` он молча проглотил
        # несовпадение сигнатуры тест-двойника, и тесты показали «сообщение не
        # отправлено» вместо ошибки вызова.
        log.info(
            "progress_not_edited",
            material_id=material_id,
            error_type=type(error).__name__,
        )
        return False
    return True


async def _show_stage(ctx: dict[str, Any], material_id: int, stage: str) -> None:
    await _edit_progress(ctx, material_id, stage_text(stage))


async def _show_done(ctx: dict[str, Any], material_id: int) -> None:
    """Сообщение о готовности. При пропущенных страницах — с их числом.

    §6.3: «пользователь не может понять, что у него на руках» — то, чего
    требование избегает. До PR B пакета WP-06 страницы-сканы пропускаются, и
    молчаливое «Готово» означало бы материал неизвестного состава. Число
    берётся из `materials.origin`, куда его положил `probe` (§5.1).
    """
    text = "✅ Готово. Материал обработан, задания в расписании."

    async with session_scope(ctx["sessions"]) as session:
        origin: dict[str, Any] | None = (
            await session.execute(select(Material.origin).where(Material.id == material_id))
        ).scalar_one()
    pending = list((origin or {}).get(VISION_PENDING_KEY) or [])
    if pending:
        text = (
            f"{text}\n\n⚠️ Страниц без текстового слоя: {len(pending)} "
            f"({', '.join(str(page) for page in pending[:10])}). "
            "Они пока не распознаны — задания по ним не созданы."
        )

    assets = list(((origin or {}).get(ASSETS_ORIGIN_KEY) or {}).get("items") or [])
    if assets:
        groups = len({item.get("group_id") for item in assets if item.get("group_id")})
        detail = f", наборов: {groups}" if groups else ""
        text = (
            f"{text}\n\n🖼 Изображений сохранено: {len(assets)}{detail}. "
            "Описания появятся, когда подключится распознавание — "
            "задания по ним пока не создаются."
        )

    await _edit_progress(ctx, material_id, text)


UNSUPPORTED_KIND_TEXT: Final[dict[str, str]] = {
    IMAGE_KIND: "Фотографии и картинки пока не обрабатываются — я умею работать с PDF и текстом.",
    AUDIO_KIND: "Голосовые и аудиофайлы пока не обрабатываются — я умею работать с PDF и текстом.",
    VIDEO_KIND: "Видео пока не обрабатывается — я умею работать с PDF и текстом.",
}
"""Что человек видит, когда адаптера для его материала ещё нет. §19.4 по духу.

Отдельно от сообщения исключения намеренно. Текст `AdapterNotReadyError`
называет пакет работ («приедет в WP-06 PR C2») — это нужно в журнале
разработчику и бессмысленно человеку: номер пакета ему ничего не говорит, да
ещё и устаревает при переносе работ. Ревью PR #58 поймало как раз показанный в
чате внутренний идентификатор вместе с питоновскими кавычками вокруг типа.

Тип материала (§3) здесь переведён на язык того, кто прислал файл:
«фотографии», а не `image`.

**Вторая карта того же соответствия**, и это названо прямо: `PENDING_KINDS`
в фабрике перечисляет типы без адаптера, здесь — текст для каждого. Вывести
один список из другого нельзя: человеческая фраза не выводится из имени типа.
Поэтому ключи взяты из тех же констант, что в фабрике, а полноту держит тест
`test_every_pending_kind_has_human_text` — без него новый тип в
`PENDING_KINDS` молча получал бы общую фразу. Урок «две карты расходятся
молча» записан в `STATE.md` после PR #42.
"""


def unsupported_kind_text(kind: str) -> str:
    """Человеческий текст отказа по типу материала.

    Неизвестный тип даёт общую фразу, а не `KeyError`: новый тип в §3 появится
    раньше, чем текст для него, и падать из-за отсутствующей строки в таблице
    значило бы менять отказ обработки на сбой системы.
    """
    return UNSUPPORTED_KIND_TEXT.get(
        kind, "Этот тип материала пока не обрабатывается — я умею работать с PDF и текстом."
    )


async def _show_failed(ctx: dict[str, Any], material_id: int, *, with_retry: bool = True) -> None:
    """Показывает причину отказа, записанную в `materials.error` (§4.5).

    Текст берётся из базы, а не из литерала здесь: его пишет тот, кто знает
    причину — `mark_failed` при сбое стадии и `expire_stale` при тайм-ауте
    §4.5, и причины разные. Литерал в этом месте был вторым источником
    истины: сообщение о тайм-ауте показывало бы текст про сбой обработки.

    Пустая причина не оставляет человека без сообщения: «понятную причину» и
    кнопку повтора требует §4.5, но если причины почему-то нет, молчание хуже
    общего текста.

    `with_retry=False` — для отказа, который повтор не исправит (§3, тип без
    адаптера): кнопка обещала бы исход, которого не будет.

    Ссылка была на §4.1 в первой редакции. Это ошибка: фраза «переводится в
    `failed` с понятной причиной; повторная обработка доступна кнопкой» стоит
    в ТЗ один раз и относится к §4.5, а §4.1 перечисляет статусы и стадии.
    Нашло ревью PR #59, и ошибка была не косметической — она же скрыла, что
    путь тайм-аута остался без проверки кнопки.
    """
    async with session_scope(ctx["sessions"]) as session:
        reason = (
            await session.execute(select(Material.error).where(Material.id == material_id))
        ).scalar_one_or_none()

    text = str(reason).strip() if reason else ""
    await _edit_progress(
        ctx,
        material_id,
        text or "Не удалось обработать материал. Нажмите «Обработать заново».",
        keyboard=retry_keyboard(material_id) if with_retry else None,
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

    # §4.5 требует «понятную причину», и причина в базе была с самого начала —
    # а сообщения не было: задача считала число и писала его в журнал.
    # Человек видел сообщение «обрабатываю…» навсегда, потому что правку
    # прогресса никто не делал. Issue #44, нашло ревью Плана.
    for material_id in expired:
        await _show_failed(ctx, material_id)

    if expired:
        log.info("awaiting_expired", count=len(expired), material_ids=expired)
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
