"""Задача воркера: путь отказа и путь ожидания. ТЗ §4.1, §4.5, §37.

Путь отказа не проверял ни один тест: ревью PR #14 вставило
`raise AssertionError` в `except Exception` и прогон не заметил. А это весь
штатный путь §4.1 `processing → failed`: перевод статуса, текст пользователю и
запись **в своей** транзакции, чтобы не быть откаченной вместе со сбоем.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from core.adapters.pdf import VISION_PENDING_KEY, PdfAdapter
from core.adapters.stub import StubAdapter
from core.db.models import Material, User
from core.db.session import session_scope
from core.ingest.awaiting import is_awaiting, park, progress_message_id, remember_message, resume
from core.ingest.handlers import ASSETS_ORIGIN_KEY
from core.storage import MaterialStorage
from tests.test_dispatcher import _settings

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


async def _material(
    engine: AsyncEngine, tg_id: int = 1200, origin: dict[str, Any] | None = None
) -> int:
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
                        kind="text",
                        status="processing",
                        mode="full",
                        completed_stages=[],
                        origin={"path": "/tmp/none.md", **(origin or {})},
                        progress={},
                        created_at=NOW,
                    )
                    .returning(Material.id)
                )
            ).scalar_one()
        )


def _ctx(sessions: async_sessionmaker[AsyncSession], tmp_path: Path) -> dict[str, Any]:
    return {
        "settings": _settings(),
        "sessions": sessions,
        "storage": MaterialStorage(tmp_path),
        "bot": None,
    }


async def _status_and_error(engine: AsyncEngine, material_id: int) -> tuple[str, str | None]:
    async with engine.connect() as connection:
        status, error = (
            await connection.execute(
                select(Material.status, Material.error).where(Material.id == material_id)
            )
        ).one()
    return str(status), error


# --- Путь отказа ----------------------------------------------------------


async def test_failed_task_marks_material_failed_with_a_reason(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой обработки переводит материал в `failed` с текстом (§4.1).

    Три свойства в одном тесте, и все три нужны: статус, текст, и то, что
    исключение поднимается дальше — иначе ARQ сочтёт задачу успешной и не
    запишет отказ в свой журнал.
    """
    import worker

    material_id = await _material(engine)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("стадия сломалась")

    monkeypatch.setattr(worker, "probe_and_gate", boom)

    with pytest.raises(RuntimeError, match="стадия сломалась"):
        await worker.process_material_task(_ctx(sessions, tmp_path), material_id, "/tmp/none.md")

    status, error = await _status_and_error(engine, material_id)
    assert status == "failed"
    assert error is not None
    assert "Обработать заново" in error, "причина не объясняет, что делать"


async def test_failure_record_survives_the_rollback(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mark_failed` пишется в своей транзакции, а не в сбойной.

    Это главное свойство пути отказа. Запись в той же границе была бы
    откачена вместе со сбоем, и материал остался бы в `processing` навсегда —
    состояния «обрабатывается, но никто не обрабатывает» §4.1 не
    предусматривает, и пользователь ждал бы вечно.

    Проверяется тем, что после исключения статус **изменился**: при записи в
    сбойной границе он остался бы прежним.
    """
    import worker

    material_id = await _material(engine)
    assert (await _status_and_error(engine, material_id))[0] == "processing"

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("сбой внутри транзакции стадии")

    monkeypatch.setattr(worker, "process_material", boom)

    with pytest.raises(RuntimeError):
        await worker.process_material_task(_ctx(sessions, tmp_path), material_id, "/tmp/none.md")

    assert (await _status_and_error(engine, material_id))[0] == "failed"


async def test_successful_task_reaches_ready(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обратное направление: при успехе материал доходит до `ready`.

    Без этого теста проверки пути отказа прошли бы и у задачи, которая падает
    всегда.
    """
    import worker
    from core.ingest import handlers as h
    from tests.llm_fakes import fake_client

    material_id = await _material(engine)
    monkeypatch.setattr(
        worker, "fragments_from_text_or_stub", lambda path: StubAdapter()._fragments
    )
    # Провайдер подменяется целиком, а не `LLM_PROVIDER` в настройках:
    # `manual` без подготовленного человеком содержания поднимает
    # `ManualContentMissingError`, и это его правильное поведение — тест о
    # пути задачи, а не о ручном режиме.
    monkeypatch.setattr(
        worker,
        "stage_handlers",
        lambda settings, sessions: h.build_handlers(fake_client(settings), settings),
    )

    result = await worker.process_material_task(
        _ctx(sessions, tmp_path), material_id, "/tmp/none.md"
    )

    assert result == "ready"
    assert (await _status_and_error(engine, material_id))[0] == "ready"


# --- message_id переживает гейт (§4.5) ------------------------------------


async def test_message_id_survives_park_and_resume(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """§4.5 «одно сообщение»: идентификатор не теряется на гейте.

    `park` и `resume` затирают `progress`, и без исключения для `message_id`
    первый же проход через гейт бюджета терял бы сообщение — прогресс после
    подтверждения уходил бы в новое. То есть «одно сообщение» нарушалось бы
    именно в сценарии, для которого ожидание и существует. Нашло ревью PR #14.
    """
    material_id = await _material(engine)

    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 4242)
        assert await progress_message_id(session, material_id) == 4242

    async with session_scope(sessions) as session:
        await park(session, material_id, "Материал большой…", kind="budget")
        assert await is_awaiting(session, material_id) is True
        assert (
            await progress_message_id(session, material_id) == 4242
        ), "park потерял идентификатор сообщения"

    async with session_scope(sessions) as session:
        await resume(session, material_id)
        assert await is_awaiting(session, material_id) is False
        assert (
            await progress_message_id(session, material_id) == 4242
        ), "resume потерял идентификатор сообщения"


async def test_park_still_clears_the_previous_question(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Сохранение `message_id` не вернуло остатки прошлого ожидания.

    `resume` затирает `question` и `payload` — иначе пользователь увидел бы
    вопрос о бюджете при выборе языка. Проверяется, что исключение сделано
    ровно для `message_id`, а не для всего `progress`.
    """
    material_id = await _material(engine)

    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 77)
        await park(session, material_id, "Про бюджет", kind="budget", payload={"a": 1})
    async with session_scope(sessions) as session:
        await resume(session, material_id)

    async with engine.connect() as connection:
        progress = (
            await connection.execute(select(Material.progress).where(Material.id == material_id))
        ).scalar_one()

    assert progress == {"message_id": 77}, f"в progress осталось лишнее: {progress}"


async def test_progress_message_id_is_none_for_seeded_material(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """`None` — нормальный исход, а не ошибка.

    Материал мог прийти через `make seed`, где Telegram не участвует вовсе.
    Падение здесь сделало бы харнесс §1.4 неработоспособным.
    """
    material_id = await _material(engine)
    async with session_scope(sessions) as session:
        assert await progress_message_id(session, material_id) is None


# --- Сообщение об отказе доходит до человека (§4.1, §4.5) -----------------


async def test_failure_message_reaches_the_user(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отказ показывается, а не только пишется в `materials.error`.

    До правки `mark_failed` ставил `progress={}`, затирая `message_id`, а
    `_show_failed` вызывается **после** него — и `_edit_progress`, не найдя
    идентификатора, молча возвращал `False`. То есть сообщение об отказе не
    доходило до человека вовсе, а тесты этого не ловили: они проверяли запись
    в базу, то есть след, а не наблюдаемое поведение. Нашло ревью Плана
    Issue #44.
    """
    import worker

    material_id = await _material(engine, tg_id=1401)
    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 555)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("стадия сломалась")

    monkeypatch.setattr(worker, "probe_and_gate", boom)

    bot = EditingBot()
    ctx = _ctx(sessions, tmp_path) | {"bot": bot}
    with pytest.raises(RuntimeError):
        await worker.process_material_task(ctx, material_id, "/tmp/none.md")

    assert bot.edited, "сообщение об отказе не отправлено"
    assert "Обработать заново" in bot.edited[-1]

    status, error = await _status_and_error(engine, material_id)
    assert status == "failed"
    assert (
        error is not None and error in bot.edited[-1]
    ), "показан не тот текст, что записан в materials.error"


async def test_failure_keeps_the_message_id_and_clears_the_rest(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
) -> None:
    """`mark_failed` сохраняет `message_id` и затирает остальной прогресс.

    Оба факта в одном тесте, и оба нужны: сохранение — чтобы сообщение было
    куда отправить (§4.5 «одно сообщение»), затирание — чтобы остатки
    ожидания прошлой попытки не выглядели ожиданием в следующей.
    """
    from core.services.materials import mark_failed, progress_of

    material_id = await _material(engine, tg_id=1402)
    async with session_scope(sessions) as session:
        await park(session, material_id, kind="budget", question="Обрабатывать?")
        await remember_message(session, material_id, 777)

    async with session_scope(sessions) as session:
        await mark_failed(session, material_id, "причина отказа")

    async with session_scope(sessions) as session:
        progress = await progress_of(session, material_id)

    assert progress == {"message_id": 777}, f"прогресс после отказа: {progress}"


async def test_timeout_message_reaches_the_user_with_its_own_reason(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§4.5: тайм-аут ожидания показывает **свою** причину, а не текст сбоя.

    До правки задача `expire_awaiting_task` только считала число и писала его
    в журнал: материал уходил в `failed`, а человек продолжал видеть
    «обрабатываю…» навсегда. Текст берётся из `materials.error`, поэтому
    причина тайм-аута и причина сбоя не путаются — литерал в `_show_failed`
    был вторым источником истины.
    """
    import worker

    material_id = await _material(engine, tg_id=1403)
    async with session_scope(sessions) as session:
        await park(session, material_id, kind="budget", question="Обрабатывать?")
        await remember_message(session, material_id, 888)
        # Ожидание началось заведомо раньше тайм-аута.
        await session.execute(
            Material.__table__.update()
            .where(Material.id == material_id)
            .values(
                progress={
                    "stage": "awaiting_user",
                    "kind": "budget",
                    "question": "Обрабатывать?",
                    "since": (NOW - datetime.timedelta(days=30)).isoformat(),
                    "message_id": 888,
                }
            )
        )

    bot = EditingBot()
    ctx = _ctx(sessions, tmp_path) | {"bot": bot}
    expired = await worker.expire_awaiting_task(ctx)

    assert expired == 1
    assert bot.edited, "сообщение о тайм-ауте не отправлено"
    assert "не получен ответ" in bot.edited[-1], f"показан не тот текст: {bot.edited[-1]!r}"


async def test_failure_without_a_progress_message_does_not_crash(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Материал из `make seed` сообщения прогресса не имеет — и это не ошибка.

    §4.5 про отображение прогресса в чате; материал, созданный консолью,
    Telegram не касается вовсе, и падать из-за косметики нельзя.
    """
    import worker

    material_id = await _material(engine, tg_id=1404)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("стадия сломалась")

    monkeypatch.setattr(worker, "probe_and_gate", boom)

    bot = EditingBot()
    ctx = _ctx(sessions, tmp_path) | {"bot": bot}
    with pytest.raises(RuntimeError):
        await worker.process_material_task(ctx, material_id, "/tmp/none.md")

    assert bot.edited == [], "правка сообщения без message_id"
    assert (await _status_and_error(engine, material_id))[0] == "failed"


# --- Выбор адаптера по materials.kind (§3, §5.1) --------------------------


def test_adapter_is_picked_by_kind_and_not_by_file_extension() -> None:
    """§3: тип материала — это `materials.kind`, а не расширение пути.

    Пара проверена накрест: для `kind = "pdf"` путь с расширением `.md`, для
    `kind = "text"` — путь с `.pdf`. При выборе по пути обе проверки
    провалились бы, при выборе по типу обе проходят.
    """
    import worker

    settings = _settings()

    assert isinstance(worker._adapter_for("pdf", "/tmp/конспект.md", settings), PdfAdapter)
    assert isinstance(worker._adapter_for("text", "/tmp/lecture.pdf", settings), StubAdapter)


async def test_task_refuses_a_kind_whose_adapter_has_not_arrived(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Тип без адаптера даёт внятный отказ, а не пустую обработку.

    Этот тест и есть доказательство, что тип читается **из базы**: путь файла
    здесь `/tmp/none.md`, то есть при выборе по расширению материал ушёл бы в
    текстовую заглушку и дошёл бы до стадий, а по `kind = "image"` фабрика
    отказывает с именем пакета, в котором адаптер приедет.
    """
    import worker

    material_id = await _material(engine, tg_id=1303)
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update().where(Material.id == material_id).values(kind="image")
        )

    with pytest.raises(ValueError, match="WP-06 PR C"):
        await worker.process_material_task(_ctx(sessions, tmp_path), material_id, "/tmp/none.md")

    assert (await _status_and_error(engine, material_id))[0] == "failed"


# --- Неподдерживаемый тип и тайм-аут по нескольким материалам -------------


async def test_unsupported_kind_fails_without_a_button(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Тип без адаптера: причина названа, кнопки нет.

    Повтор такого материала привёл бы к тому же отказу, пока адаптер не
    приедет, — поэтому текст называет причину и **не** предлагает повтор.
    Различается **типом** исключения (`AdapterNotReadyError`), а не разбором
    текста сообщения: разбор был бы п. 7 списка костылей.

    Кнопка приезжает следующим PR; здесь проверяется, что текст про повтор
    этому отказу не достаётся.
    """
    import worker

    material_id = await _material(engine, tg_id=1502)
    async with engine.begin() as connection:
        await connection.execute(
            Material.__table__.update().where(Material.id == material_id).values(kind="image")
        )
        await connection.execute(
            Material.__table__.update().where(Material.id == material_id).values(progress={})
        )
    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 102)

    bot = EditingBot()
    with pytest.raises(ValueError, match="WP-06 PR C"):
        await worker.process_material_task(
            _ctx(sessions, tmp_path) | {"bot": bot}, material_id, "/tmp/none.md"
        )

    assert bot.edited, "сообщение не отправлено"
    assert "пока не поддерживается" in bot.edited[-1]
    assert (
        "Обработать заново" not in bot.edited[-1]
    ), "предложен повтор на отказе, который повтор не исправит"


async def test_timeout_reaches_each_material_with_its_own_message(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Два материала — два сообщения, каждому своё. Замечание ревью PR #57.

    Один материал не отличал «у каждого свой `message_id`» от «взяли первый
    попавшийся»: `expire_stale` переводит в `failed` несколько материалов за
    раз, и отправка обязана идти по каждому.
    """
    import worker

    stale = {
        "stage": "awaiting_user",
        "kind": "budget",
        "question": "Обрабатывать?",
        "since": (NOW - datetime.timedelta(days=30)).isoformat(),
    }
    for tg_id, message_id in ((1503, 201), (1504, 202)):
        material_id = await _material(engine, tg_id=tg_id)
        async with session_scope(sessions) as session:
            await session.execute(
                Material.__table__.update()
                .where(Material.id == material_id)
                .values(progress={**stale, "message_id": message_id})
            )

    bot = EditingBot()
    expired = await worker.expire_awaiting_task(_ctx(sessions, tmp_path) | {"bot": bot})

    assert expired == 2
    assert len(bot.edited) == 2, f"сообщений {len(bot.edited)}, а материалов 2"
    assert all("не получен ответ" in text for text in bot.edited)


# --- Сообщение о готовности называет пропущенные страницы (§6.3) ----------


class EditingBot:
    """Записывает правки текста сообщения прогресса.

    Подставляется в `ctx["bot"]` напрямую: проверяется, что печатает воркер, а
    не как aiogram доставляет правку.
    """

    def __init__(self) -> None:
        self.edited: list[str] = []

    async def edit_message_text(self, *, text: str, chat_id: int, message_id: int) -> None:
        self.edited.append(text)


async def _done_text(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    origin: dict[str, Any] | None,
    tg_id: int,
) -> str:
    import worker

    material_id = await _material(engine, tg_id=tg_id, origin=origin)
    async with session_scope(sessions) as session:
        await remember_message(session, material_id, 777)

    bot = EditingBot()
    ctx = _ctx(sessions, tmp_path) | {"bot": bot}
    await worker._show_done(ctx, material_id)

    assert bot.edited, "сообщение о готовности не отправлено вовсе"
    return bot.edited[-1]


async def test_done_message_names_the_pages_awaiting_vision(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§6.3: последствия пропуска страниц видны пользователю, а не только в базе.

    Третье из трёх мест видимости `vision_pending_pages` — сообщение в чате
    (первые два: `materials.origin` и `make dump-material`). До этого теста
    ветвь не держалась ничем: живой прогон PR A шёл через `make seed`, где
    Telegram не участвует, а `worker.py` не входит и в `files` для `mypy`.

    Пара ассертов, а не один: без второго теста — на материале без пропусков —
    проверка прошла бы и у реализации, которая печатает предупреждение всегда.
    """
    with_pending = await _done_text(
        engine, sessions, tmp_path, {VISION_PENDING_KEY: [1, 3]}, tg_id=1301
    )

    assert "Готово" in with_pending
    assert "Страниц без текстового слоя: 2" in with_pending
    assert "1, 3" in with_pending, "номера пропущенных страниц не названы"


async def test_done_message_names_saved_images(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """§6.3: сохранённые изображения названы, пока описаний у них нет.

    До C2 ассет существует без описания, то есть без фрагмента: человеку
    нужно видеть, что картинки сохранены, но заданий по ним ещё не будет.
    Число наборов называется отдельно — набор это единица, которую потребляет
    `match_image` (§9.2).
    """
    text = await _done_text(
        engine,
        sessions,
        tmp_path,
        {
            ASSETS_ORIGIN_KEY: {
                "items": [
                    {"asset_id": 1, "page": 10, "group_id": "page10-fig1"},
                    {"asset_id": 2, "page": 10, "group_id": "page10-fig1"},
                    {"asset_id": 3, "page": 3, "group_id": None},
                ]
            }
        },
        tg_id=1304,
    )

    assert "Изображений сохранено: 3" in text
    assert "наборов: 1" in text


async def test_done_message_says_nothing_about_images_when_there_are_none(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Обратное направление: без ассетов про изображения не говорится.

    Иначе предыдущий тест прошёл бы и у реализации, которая печатает строку
    всегда — с нулём.
    """
    text = await _done_text(engine, sessions, tmp_path, None, tg_id=1305)

    assert "Изображений" not in text


async def test_done_message_is_plain_when_nothing_was_skipped(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
) -> None:
    """Обратное направление: без пропусков предупреждения нет."""
    plain = await _done_text(engine, sessions, tmp_path, None, tg_id=1302)

    assert "Готово" in plain
    assert "текстового слоя" not in plain
