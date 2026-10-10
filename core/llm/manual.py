"""Провайдер `manual`: содержание готовит человек. ТЗ §2.1, §34.1.

Спайк §34.1 не выполнялся — ключа провайдера нет, — а продукт нужен рабочим на
одном пользователе. Решение Архитектора: режим `manual` соблюдает контракт
`LLMProvider` целиком, **токены считает оценкой**, стоимость даёт нулём, и
строку в `llm_calls` пишет как любой вызов (§2.1). Причину нулевой стоимости
стоит помнить: вызова действительно не было, и ненулевая сумма означала бы
расход, которого нет, — а вот нулевые **токены** скрыли бы объём работы,
прошедшей через слой за весь период ручного режима.

**Как это работает.** Провайдер ищет готовый ответ в файле, адрес которого
определяется ключом из `core/llm/recording.py` — тем же, каким пользуется
режим `replay`. Нашёл — разбирает по схеме и возвращает. Не нашёл — пишет
рядом **файл-запрос** с промптом и схемой и поднимает
`ManualContentMissingError`.

Файл-запрос это не отладочный вывод, а рабочий механизм режима: без него
человек видит «нет содержания для ключа 9f3c…» и не знает ни что именно
спрашивали, ни в каком виде отвечать, а ключ — хеш, и восстановить по нему
промпт нельзя. Поэтому запрос пишется на диск намеренно, и это единственное
место слоя, где чтение имеет побочный эффект.

**Счётчики всегда оценочные, даже когда в файле есть настоящие.** Запись
`tests/golden` хранит токены машинного вызова, но `manual` их не присваивает:
вызова не было, и выдать записанные числа за свои значило бы утверждать, что
модель отвечала сейчас. Настоящие счётчики записи используется режимом
`replay` (`core/llm/recording.load`), где это уместно, — там воспроизводится
именно тот вызов, который их потратил.

**Два каталога, и порядок между ними важен.** Сначала
`data/materials/manual` — том §30.1, куда ответы кладёт человек; затем
`tests/golden` — записи в репозитории. Такой порядок позволяет перекрыть
запись руками, не правя репозиторий; обратный заставлял бы удалять фикстуру,
чтобы испытать новый ответ.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core.llm.base import ImageInput, LLMProvider, LLMResult, SchemaError, TokenUsage
from core.llm.estimation import tokens_in
from core.llm.recording import GOLDEN_ROOT, recording_key, recording_path
from core.logging import get_logger

log = get_logger(__name__)

MANUAL_ROOT = Path("data/materials/manual")
"""Пятый том §30.1 (CR-G). Содержание, подготовленное руками, — это данные
пользователя, и в репозитории ему не место: том переживает пересборку образа,
а `tests/golden` живёт в git, потому что участвует в проверках."""

REQUEST_SUFFIX = ".request.md"
"""Расширение файла-запроса. Markdown, а не JSON: его читает человек."""

MODEL_LABEL = "manual"
"""Значение `llm_calls.model` (§3).

Метка режима, а не имя модели. Написать сюда `claude-opus-5` значило бы
утверждать, что модель отвечала, и §6.4 показывал бы историю вызовов, которых
не было.
"""

ENVELOPE_KEYS = ("value", "model")
"""Признак машинной записи `core/llm/recording.py`.

Файл с этими ключами разбирается как конверт, иначе — как сам ответ. Различие
нужно потому, что человек пишет ответ прямо, без обёртки, а `tests/golden`
хранит конверт со счётчиками. Проверяются **два** ключа: ответ с собственным
полем `value` встречается легко, ответ, у которого есть и `value`, и `model`,
— практически нет.
"""


class ManualContentMissingError(Exception):
    """Готового ответа для этого вызова нет.

    Не наследник `ProviderUnavailableError`: повтор бессмысленен, пока человек
    не положит файл, и экспоненциальный backoff здесь означал бы четыре
    ожидания впустую. Обработка материала останавливается, и это штатный ход
    ручного режима, а не сбой слоя.
    """


def _with_images(prompt: str, images: Sequence[ImageInput]) -> str:
    """Добавляет к промпту список файлов изображений.

    Нужно для двух вещей сразу. Человеку — знать, какие страницы открывать.
    Ключу — зависеть от изображений: без этого две разные страницы с одним
    промптом получили бы один ответ, и вторая страница молча вернула бы
    содержание первой.
    """
    listing = "\n".join(f"- {image.path} ({image.mime})" for image in images)
    return f"{prompt}\n\nИзображения:\n{listing}"


def _estimate(payload: str, value: BaseModel) -> TokenUsage:
    """Оценка токенов по длине входа и выхода (§2.1, решение Архитектора).

    `estimated=True` — обязательно: §6.4 сравнивает оценку с фактом, и
    пометка отличает одно от другого. Без неё ручной период выглядел бы как
    идеальное совпадение оценки с фактом, то есть коридор §6.4 показывал бы
    ноль расхождения там, где он просто нечего не сравнивал.
    """
    answer = json.dumps(value.model_dump(mode="json"), ensure_ascii=False)
    return TokenUsage(
        input_tokens=tokens_in(payload),
        output_tokens=tokens_in(answer),
        estimated=True,
    )


class ManualProvider(LLMProvider):
    """Читает готовые ответы с диска. §2.1, режим `LLM_PROVIDER=manual`."""

    name = MODEL_LABEL

    offline = True
    """Сети не касается, поэтому клиент не подменяет его вызовы в режиме
    `replay`: подменять нечего, а перехват скрыл бы содержание, подготовленное
    руками на томе §30.1."""

    def __init__(self, roots: Sequence[Path] | None = None) -> None:
        self._roots = tuple(roots) if roots is not None else (MANUAL_ROOT, GOLDEN_ROOT)
        if not self._roots:
            raise ValueError(
                "ManualProvider без каталогов поиска: искать ответы было бы негде, "
                "и каждый вызов поднимал бы ManualContentMissingError"
            )

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, purpose: str
    ) -> LLMResult[T]:
        return self._answer(prompt, schema, purpose=purpose)

    async def vision[T: BaseModel](
        self,
        images: list[ImageInput],
        prompt: str,
        schema: type[T],
        *,
        purpose: str,
    ) -> LLMResult[T]:
        return self._answer(_with_images(prompt, images), schema, purpose=purpose)

    def _answer[T: BaseModel](self, payload: str, schema: type[T], *, purpose: str) -> LLMResult[T]:
        key = recording_key(payload, schema)
        for root in self._roots:
            path = recording_path(purpose, key, root=root)
            if path.is_file():
                value = _parse(path, schema)
                log.info("manual_answer_used", purpose=purpose, key=key, path=str(path))
                return LLMResult(value=value, usage=_estimate(payload, value), model=MODEL_LABEL)

        raise ManualContentMissingError(self._request(payload, schema, purpose=purpose, key=key))

    def _request(self, payload: str, schema: type[BaseModel], *, purpose: str, key: str) -> str:
        """Пишет файл-запрос и возвращает текст исключения.

        Текст исключения называет **путь ответа**, а не путь запроса: человек
        прочитает запрос и положит ответ, и первым ему нужен адрес ответа.
        """
        answer_path = recording_path(purpose, key, root=self._roots[0])
        request_path = answer_path.with_suffix(REQUEST_SUFFIX)
        body = (
            f"# Запрос к человеку: {purpose}\n\n"
            f"- ключ: `{key}`\n"
            f"- ответ положить в: `{answer_path}`\n"
            f"- формат ответа: JSON по схеме ниже, без обёртки\n\n"
            f"## Схема ответа\n\n```json\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}\n"
            f"```\n\n## Промпт\n\n{payload}\n"
        )
        try:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(body, encoding="utf-8")
        except OSError as error:
            # Запрос не записался — чаще всего нет прав на том. Исключение
            # всё равно поднимается, но уже без адреса, по которому человек
            # прочтёт промпт, и об этом надо сказать прямо: иначе он пойдёт
            # искать несуществующий файл.
            log.warning("manual_request_not_written", path=str(request_path), error=str(error))
            return (
                f"нет готового ответа для {purpose}/{key}; ответ ожидается в {answer_path}. "
                f"Файл-запрос записать не удалось ({error}), промпт есть только в логе."
            )

        log.info("manual_request_written", purpose=purpose, key=key, path=str(request_path))
        return (
            f"нет готового ответа для {purpose}/{key}. "
            f"Промпт и схема — в {request_path}, ответ положить в {answer_path}."
        )


def _parse[T: BaseModel](path: Path, schema: type[T]) -> T:
    """Разбирает файл ответа: конверт записи или сам ответ.

    Разбор не переиспользует `recording.load`: тот строго требует конверт со
    счётчиками, и это правильно для машинных записей — запись без токенов
    сделала бы воспроизведённый вызов бесплатным. Ответ человека счётчиков
    не содержит и содержать не может, поэтому здесь свой разбор, а строгость
    `load` остаётся нетронутой.
    """
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and all(field in raw for field in ENVELOPE_KEYS):
        raw = raw["value"]

    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise SchemaError(
            f"ответ {path} не разбирается по схеме {schema.__name__}: {error}"
        ) from error
