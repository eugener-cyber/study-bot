"""Машина стадий. ТЗ §4.1, §4.2, §4.3.

Конвейер идёт по перечню §4.1, начиная с первой незавершённой стадии, и каждую
выполняет в собственной транзакционной границе. Порядок внутри границы задан
§4.2 и воспроизводится буквально:

```
BEGIN
  подготовить все артефакты стадии
  проверить артефакты
  удалить предыдущие частичные артефакты этой стадии
  сохранить полный результат
  добавить stage в completed_stages
COMMIT
```

Этот порядок — не стиль. Удаление предыдущих частичных артефактов стоит
**после** проверки, то есть старый мусор не сносится, пока не известно, что
новый результат годен. Внутри одной транзакции разница незаметна, но она
становится видимой, если стадия когда-нибудь начнёт писать частями, и
соблюдать порядок дешевле, чем потом искать, почему он был другим.

**Про отметку в `completed_stages` и порядок вызовов.** Сейчас безразлично,
ставится она до вызова обработчика или после: оба действия в одной транзакции
и при сбое откатываются вместе. Мутационный прогон это подтвердил, и
Проверяющий назвал условие, при котором эквивалентность исчезает: **как только
обработчик начнёт читать `completed_stages`**, порядок станет существенным, и
отметка до вызова покажет стадию выполненной тому самому коду, который её
выполняет. Сегодня этого не делает ни один обработчик, и ни один не вызывает
`commit`/`rollback` — транзакцией распоряжается только `core/db/session.py`.

Файловая часть атомарности — в `core/storage.py`: база и диск не делят
транзакцию, и `os.rename` не откатывается.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.adapters.base import SourceAdapter
from core.db.models import Material
from core.db.session import session_scope
from core.ingest.stages import ORDER, Stage, is_done, next_stage
from core.logging import get_logger
from core.storage import MaterialStorage

log = get_logger(__name__)


class AwaitingUser(Exception):  # noqa: N818
    """Стадия требует ответа пользователя и останавливает конвейер.

    Не ошибка, поэтому и не `Error` в имени. §4.5: «материал остаётся в статусе
    `processing` с `progress.stage = 'awaiting_user'`, задача ARQ завершается,
    а продолжение ставится в очередь обработчиком нажатия кнопки. Фоновый
    воркер никогда не блокируется в ожидании ответа».

    Исключение выбрано как механизм выхода намеренно: альтернатива — особое
    возвращаемое значение, которое каждый вызывающий обязан проверить. Забытая
    проверка дала бы молчаливое продолжение конвейера без ответа пользователя,
    то есть обход гейта §6.3.
    """

    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question


@dataclass(frozen=True)
class StageContext:
    """Что стадия получает, но не формирует сама."""

    material_id: int
    tmp_dir: Path
    adapter: SourceAdapter
    source_path: str


class StageHandler(Protocol):
    """Реализация одной стадии.

    Обязана соблюдать внутренний порядок §4.2: подготовить, проверить, удалить
    предыдущие частичные артефакты этой стадии, сохранить результат. Отметку в
    `completed_stages` ставит конвейер — стадия этого не делает, иначе
    появились бы шесть мест, где материал объявляется продвинувшимся.
    """

    async def __call__(
        self, session: AsyncSession, material: Material, ctx: StageContext
    ) -> None: ...


Handlers = dict[str, StageHandler]


async def _load(session: AsyncSession, material_id: int) -> Material:
    return (await session.execute(select(Material).where(Material.id == material_id))).scalar_one()


async def _is_notes_only(sessions: async_sessionmaker[AsyncSession], material_id: int) -> bool:
    """Режим материала читается из базы на каждой стадии, а не кешируется.

    Между стадиями пользователь мог нажать «Только конспект, без тестов» или,
    наоборот, «Достроить задания» (§6.3), и значение, прочитанное один раз в
    начале, относилось бы к прошлому решению.
    """
    async with session_scope(sessions) as session:
        mode = (
            await session.execute(select(Material.mode).where(Material.id == material_id))
        ).scalar_one()
    return mode == "notes_only"


async def _mark_completed(session: AsyncSession, material_id: int, marker: str) -> None:
    """Добавляет отметку в `completed_stages` внутри транзакции стадии.

    Через `array_append` в SQL, а не чтением и записью списка в Python: второй
    способ теряет отметку, если тот же материал параллельно продвинула другая
    задача. Повторный запуск воркера после сбоя — ровно такой случай.
    """
    await session.execute(
        update(Material)
        .where(Material.id == material_id)
        .values(completed_stages=Material.completed_stages.op("||")([marker]))
    )


async def run_stage(
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    handler: StageHandler,
    stage: Stage,
    ctx_factory: Callable[[Path], StageContext],
    material_id: int,
) -> bool:
    """Выполняет одну стадию в своей транзакционной границе.

    Возвращает `True`, если стадия выполнена, и `False`, если была уже
    выполнена раньше (§4.3: повторный запуск завершённой стадии — no-op).

    `AwaitingUser` пропускается наружу без отметки о выполнении: материал
    остаётся на этой стадии и продолжит с неё же после ответа пользователя.
    """
    tmp_dir = storage.prepare_tmp(material_id, stage.name)
    ctx = ctx_factory(tmp_dir)

    try:
        async with session_scope(sessions) as session:
            material = await _load(session, material_id)
            if is_done(stage, list(material.completed_stages or [])):
                storage.discard_tmp(material_id, stage.name)
                return False

            await handler(session, material, ctx)
            await _mark_completed(session, material_id, stage.name)
    except Exception:
        # Транзакция откачена в `session_scope`; здесь убирается файловая
        # часть. Порядок важен: если убрать файлы до коммита, успешная стадия
        # потеряла бы артефакты.
        storage.discard_tmp(material_id, stage.name)
        raise

    storage.commit_tmp(material_id, stage.name)
    log.info("stage_completed", stage=stage.name, material_id=material_id)
    return True


async def process_material(
    sessions: async_sessionmaker[AsyncSession],
    storage: MaterialStorage,
    handlers: Handlers,
    material_id: int,
    adapter: SourceAdapter,
    source_path: str,
    *,
    stages: Sequence[Stage] = ORDER,
) -> None:
    """Доводит материал до конца конвейера, начиная с первой незавершённой.

    §4.3: «Воркер читает `completed_stages` и начинает с первой незавершённей
    стадии». Перечень перечитывается из базы на каждой стадии, а не берётся
    один раз: между стадиями мог вмешаться ответ пользователя, переведший
    материал в `notes_only`, и список стадий изменился бы.
    """
    storage.cleanup_material_tmp(material_id)

    while True:
        async with session_scope(sessions) as session:
            material = await _load(session, material_id)
            completed = list(material.completed_stages or [])

        stage = next_stage(completed)
        if stage is None or stage not in stages:
            break

        if stage.skippable and await _is_notes_only(sessions, material_id):
            # §6.3, режим «Только конспект, без тестов». Пропуск фиксируется
            # как `questions:skipped`, а не `questions`: по этой разнице
            # возобновление отличает «пропущено сознательно» от «выполнено», а
            # кнопка «Достроить задания» знает, что доделывать (§4.1).
            async with session_scope(sessions) as session:
                await _mark_completed(session, material_id, stage.skipped_marker)
            log.info("stage_skipped", stage=stage.name, material_id=material_id)
            continue

        handler = handlers.get(stage.name)
        if handler is None:
            raise LookupError(f"нет обработчика для стадии {stage.name}")

        def make_ctx(tmp_dir: Path, _stage: Stage = stage) -> StageContext:
            return StageContext(
                material_id=material_id,
                tmp_dir=tmp_dir,
                adapter=adapter,
                source_path=source_path,
            )

        await run_stage(sessions, storage, handler, stage, make_ctx, material_id)


def pending_stages(completed: list[str]) -> tuple[Stage, ...]:
    """Что осталось выполнить. Для сообщения прогресса (§4.5)."""
    return tuple(stage for stage in ORDER if not is_done(stage, completed))


StageRunner = Callable[..., Awaitable[None]]
