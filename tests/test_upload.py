"""Приём материала. ТЗ §2.2, §4.4, §31, §26."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Document, Message, PhotoSize
from aiogram.types import User as TgUser
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from bot.handlers.upload import NO_CONSENT, UNSUPPORTED_KIND, _kind_of, handle_material
from core.adapters.factory import BY_EXTENSION, IMAGE_KIND, PENDING_KINDS, kind_for_path
from core.db.models import Material, User
from core.db.session import session_scope

pytestmark = pytest.mark.db

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _message(**extra: object) -> Message:
    message = Message(
        message_id=1,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
        from_user=TgUser(id=900, is_bot=False, first_name="T"),
        **extra,  # type: ignore[arg-type]
    )
    reply = Message(
        message_id=2,
        date=datetime.datetime(2026, 1, 1),
        chat=Chat(id=1, type="private"),
    )
    object.__setattr__(reply, "edit_text", AsyncMock())
    object.__setattr__(message, "answer", AsyncMock(return_value=reply))
    return message


def _document(name: str) -> Document:
    return Document(file_id="fid", file_unique_id="uid", file_name=name, file_size=10)


async def _consented_user(engine: AsyncEngine, tg_id: int = 900) -> int:
    async with engine.begin() as connection:
        return int(
            (
                await connection.execute(
                    insert(User)
                    .values(tg_id=tg_id, created_at=NOW, consent_at=NOW)
                    .returning(User.id)
                )
            ).scalar_one()
        )


# --- Тип источника --------------------------------------------------------


def test_pdf_document_is_recognised() -> None:
    assert _kind_of(_message(document=_document("lecture.pdf"))) == "pdf"


def test_markdown_and_text_are_recognised() -> None:
    """Текстовые файлы принимаются по-настоящему, а не заглушкой.

    Это то, на чём работает ручная подготовка содержания для провайдера
    `manual` (§2.1, CR-H): конспект в Markdown проходит конвейер целиком.
    """
    assert _kind_of(_message(document=_document("конспект.md"))) == "text"
    assert _kind_of(_message(document=_document("notes.txt"))) == "text"


def test_photo_is_recognised_as_image() -> None:
    photo = [PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)]
    assert _kind_of(_message(photo=photo)) == "image"


def test_unsupported_document_gives_none_not_error() -> None:
    """`None` означает «не поддерживается здесь», а не ошибку.

    Аудио и видео появятся в WP-19 и WP-20, ссылки в WP-22. До тех пор ответ о
    них должен быть внятным, а не исключением.
    """
    assert _kind_of(_message(document=_document("lecture.mp3"))) is None
    assert _kind_of(_message(text="просто текст")) is None


def test_kind_of_a_photo_matches_what_the_factory_gives_for_the_saved_name() -> None:
    """Приём и фабрика выводят тип по одной карте. Замечание ревью PR #42.

    Фотография сохраняется под именем `photo-<id>.jpg` (ниже в этом же
    хендлере), и до правки она получала `image` здесь и `text` в фабрике — то
    есть воркер выбирал адаптер не для того типа, который записан в базе.
    Тест сравнивает два вывода напрямую: при расхождении он краснеет, а не
    ждёт появления адаптера изображений в PR C.
    """
    photo = [PhotoSize(file_id="ab12cd34", file_unique_id="u", width=1, height=1)]

    assert _kind_of(_message(photo=photo)) == kind_for_path("photo-ab12cd34.jpg")


def test_kind_of_a_document_matches_the_factory_map() -> None:
    """То же для документов: тип берётся из `BY_EXTENSION`, а не выводится заново."""
    for name in ("lecture.pdf", "конспект.md", "notes.txt"):
        assert _kind_of(_message(document=_document(name))) == BY_EXTENSION[Path(name).suffix]


def test_image_sent_as_a_document_is_refused_until_its_adapter_arrives() -> None:
    """Принимается только то, что уже умеет обрабатываться. Ревью PR #42, фаза 2.

    `.jpg` документом — честный отказ §26 (`UNSUPPORTED_KIND`), а не материал
    `image`, который дойдёт до воркера и упадёт на выборе адаптера: человек
    получил бы отказ обработки вместо отказа приёма, то есть «сломалось»
    вместо «пока не умею».

    Полоса привязана к `PENDING_KINDS`, а не к списку расширений: когда
    адаптер изображений приедет в PR C, тип уйдёт из `PENDING_KINDS` и приём
    откроется сам — и этот тест тогда обязан покраснеть, а не промолчать.
    """
    for name in ("photo.jpg", "scan.JPEG", "page.png", "shot.webp"):
        assert _kind_of(_message(document=_document(name))) is None, name

    assert BY_EXTENSION[".jpg"] in PENDING_KINDS, "полоса держится не на том признаке"


def test_photo_is_still_accepted_although_its_adapter_has_not_arrived() -> None:
    """Обратная сторона: фотографию §26 принимает, и это поведение до пакета.

    Иначе предыдущий тест прошёл бы и у реализации, которая закрыла приём
    изображений целиком — а §26 требует принимать присланное фото.
    """
    photo = [PhotoSize(file_id="f", file_unique_id="u", width=1, height=1)]

    assert _kind_of(_message(photo=photo)) == IMAGE_KIND


# --- Согласие -------------------------------------------------------------


async def test_file_is_not_accepted_without_consent(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Материал нельзя принять раньше согласия (§31).

    Приём означает хранение файла, то есть начало обработки персональных
    данных, а согласие — ровно то, что это разрешает. Проверяется и ответ, и
    отсутствие строки материала: ответ без второго ассерта прошёл бы и у
    обработчика, который принял файл и заодно попросил согласие.
    """
    message = _message(document=_document("lecture.pdf"))
    async with session_scope(sessions) as session:
        await handle_material(message, session)

    message.answer.assert_awaited_once_with(NO_CONSENT)  # type: ignore[attr-defined]
    async with engine.connect() as connection:
        assert (await connection.execute(select(Material.id))).all() == []


async def test_file_is_not_downloaded_without_consent(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Проверка согласия идёт **до** скачивания.

    Обратный порядок означал бы, что файл человека без согласия уже лежит на
    диске. Проверяется тем, что `bot` не понадобился вовсе: скачивание первым
    действием обратилось бы к нему и упало.
    """
    message = _message(document=_document("lecture.pdf"))
    assert message.bot is None, "подготовка теста: бот не привязан"

    async with session_scope(sessions) as session:
        await handle_material(message, session)

    message.answer.assert_awaited_once_with(NO_CONSENT)  # type: ignore[attr-defined]


# --- Неподдерживаемый тип -------------------------------------------------


async def test_unsupported_kind_gets_an_answer_not_silence(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Аудио получает честный отказ, а не тишину.

    Это то, чем заменён переходный §19.4 (CR-F, Issue #5): там бот молчал,
    потому что не умел принимать ничего. Теперь он умеет принимать часть и
    говорит, какую именно.
    """
    await _consented_user(engine)
    message = _message(document=_document("lecture.mp3"))

    async with session_scope(sessions) as session:
        await handle_material(message, session)

    message.answer.assert_awaited_once_with(UNSUPPORTED_KIND)  # type: ignore[attr-defined]
    assert "Аудио" in UNSUPPORTED_KIND, "отказ не называет, чего именно нет"


async def test_unsupported_kind_names_what_is_accepted(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], clean_tables: None
) -> None:
    """Отказ говорит, что бот принимает **сейчас**.

    Без этого ответ «пока не умею» оставляет человека угадывать, что
    попробовать следующим.
    """
    assert "документ" in UNSUPPORTED_KIND.lower()
    assert "фотограф" in UNSUPPORTED_KIND.lower()


# --- Дубликат (§4.4) ------------------------------------------------------


async def test_duplicate_file_does_not_create_second_material(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.4: дубль показывает существующий материал, копия не создаётся.

    Скачивание подменяется: проверяется поведение при дубликате, а не работа
    локального Bot API. Файл один и тот же, поэтому `sha256` совпадает.
    """
    import bot.handlers.upload as upload_module

    await _consented_user(engine)
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    async def fake_download(message: object, file_id: str, suffix: str) -> Path:
        return source

    monkeypatch.setattr(upload_module, "_download", fake_download)

    first = _message(document=_document("lecture.pdf"))
    async with session_scope(sessions) as session:
        await handle_material(first, session)

    second = _message(document=_document("lecture.pdf"))
    async with session_scope(sessions) as session:
        await handle_material(second, session)

    async with engine.connect() as connection:
        rows = (await connection.execute(select(Material.id))).all()
    assert len(rows) == 1, "дубль создал второй материал"

    reply = second.answer.return_value  # type: ignore[attr-defined]
    text = reply.edit_text.await_args.args[0]
    assert "уже есть" in text


async def test_material_is_queued_when_worker_is_absent(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без воркера материал остаётся в `queued`, а не обрабатывается в хендлере.

    §4.5 требует, чтобы конвейер шёл фоном. Обработка в хендлере блокировала
    бы ответы пользователю на минуты, и это был бы не недосмотр, а другая
    архитектура.
    """
    import bot.handlers.upload as upload_module

    await _consented_user(engine)
    source = tmp_path / "notes.md"
    source.write_text("Абзац.\n\nВторой абзац.\n", encoding="utf-8")

    async def fake_download(message: object, file_id: str, suffix: str) -> Path:
        return source

    monkeypatch.setattr(upload_module, "_download", fake_download)

    message = _message(document=_document("notes.md"))
    async with session_scope(sessions) as session:
        await handle_material(message, session, arq=None)

    async with engine.connect() as connection:
        status, kind = (await connection.execute(select(Material.status, Material.kind))).one()
    assert status == "queued"
    assert kind == "text"


async def test_job_is_enqueued_when_worker_is_present(
    engine: AsyncEngine,
    sessions: async_sessionmaker[AsyncSession],
    clean_tables: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """С воркером задача ставится в очередь с `material_id` и путём.

    Положительное направление к предыдущему тесту: проверка только «без
    воркера остаётся queued» прошла бы и у обработчика, который не ставит
    задачу никогда.
    """
    import bot.handlers.upload as upload_module

    await _consented_user(engine)
    source = tmp_path / "notes.md"
    source.write_text("Абзац.\n", encoding="utf-8")

    async def fake_download(message: object, file_id: str, suffix: str) -> Path:
        return source

    monkeypatch.setattr(upload_module, "_download", fake_download)

    enqueued: list[tuple[str, tuple[object, ...]]] = []

    class FakeArq:
        async def enqueue_job(self, name: str, *args: object) -> None:
            enqueued.append((name, args))

    message = _message(document=_document("notes.md"))
    async with session_scope(sessions) as session:
        await handle_material(message, session, FakeArq())  # type: ignore[arg-type]

    assert len(enqueued) == 1
    name, args = enqueued[0]
    assert name == "process_material_task"
    assert args[1] == str(source), "в задачу ушёл не тот путь"
