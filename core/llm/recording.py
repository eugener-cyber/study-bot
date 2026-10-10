"""Режимы record / replay / live. План реализации §1.5.

```
LLM_MODE=record   реальный вызов, ответ пишется в tests/golden/<purpose>/<hash>.json
LLM_MODE=replay   значение по умолчанию, сеть недоступна, ответ берётся из файла
LLM_MODE=live     только под @pytest.mark.live
```

«Тесты не ходят в сеть, но моки нельзя писать руками — сочинённый JSON не
похож на реальный вывод модели, и ошибки промптов прячутся до продакшена».

**Ключ записи — хеш от (готовый промпт, схема).** §1.5: «Смена версии промпта
автоматически делает старые записи недействительными, и это видно по падению
тестов, а не по странному поведению». Версия участвует через сам текст: загрузчик
`core/llm/prompts.py` вставляет в начало промпта строку с версией, так что правка
версии меняет текст, а значит и ключ. Отдельным параметром версия не передаётся
намеренно — провайдер видит только готовый промпт (§2.1), и два способа считать
один ключ разошлись бы: клиент искал бы запись по одному адресу, провайдер
`manual` по другому. Схема входит в ключ потому, что, изменив поля ответа, мы
меняем задачу, и старая запись ей больше не отвечает.

`replay` — значение по умолчанию, и это защита: тест, забывший подменить
провайдера, не уйдёт в сеть, а упадёт на отсутствующей фикстуре с указанием
ожидаемого пути.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from core.llm.base import LLMResult, SchemaError, TokenUsage
from core.logging import get_logger

log = get_logger(__name__)

GOLDEN_ROOT = Path("tests/golden")
"""Записанные ответы — в репозитории: они часть проверок, а не данные.

Содержание, подготовленное вручную для провайдера `manual`, живёт отдельно, в
`data/materials/manual` на томе §30.1. Это разные вещи: первое машинное и
воспроизводимое, второе — материалы пользователя.
"""


class LLMMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"
    LIVE = "live"


class RecordingMissingError(Exception):
    """Фикстуры для ключа нет.

    Сообщение называет **ожидаемый путь**, а не только факт отсутствия: без
    пути отладка начинается с поиска, где именно искали, а ключ — это хеш, и
    угадать его нельзя.
    """


def recording_key(prompt: str, schema: type[BaseModel]) -> str:
    """Ключ записи: хеш от готового промпта и схемы ответа.

    Схема участвует как `model_json_schema()` целиком. Это значит, что ключ
    меняется и от состава полей, и от **имени класса**: pydantic кладёт имя в
    `title`. Переименование схемы, таким образом, обесценивает все её записи,
    хотя задачу не меняет.

    Выбран этот вариант, а не вычистка `title` из схемы перед хешированием.
    Цена ошибки несимметрична: лишнее обесценивание записи стоит одной
    перезаписи, а слияние ключей двух разных схем вернуло бы ответ **не на тот
    вопрос** — и вернуло бы молча, потому что ответ разобрался бы по схеме.
    Рекурсивная чистка вложенных схем — именно тот код, чья ошибка даёт второе.

    Промпт участвует целиком, вместе с подставленным в него текстом материала:
    записанный ответ относится к конкретному входу, и ключ без входа означал бы
    один ответ на все фрагменты.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(schema.model_json_schema(), sort_keys=True).encode())
    digest.update(b"\x00")
    digest.update(prompt.encode())
    return digest.hexdigest()[:16]


def recording_path(purpose: str, key: str, *, root: Path | None = None) -> Path:
    return (root or GOLDEN_ROOT) / purpose / f"{key}.json"


def save[T: BaseModel](
    purpose: str,
    key: str,
    result: LLMResult[T],
    *,
    root: Path | None = None,
) -> Path:
    """Пишет ответ провайдера на диск. Режим `record`.

    Записываются и счётчики: §6.4 сравнивает оценку с фактом, и запись без
    токенов сделала бы воспроизведённый вызов бесплатным — то есть учёт
    стоимости в режиме `replay` показывал бы нули при реально потраченных
    деньгах на записи.
    """
    path = recording_path(purpose, key, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": result.model,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "value": result.value.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log.info("recording_saved", purpose=purpose, key=key)
    return path


def load[T: BaseModel](
    purpose: str,
    key: str,
    schema: type[T],
    *,
    root: Path | None = None,
) -> LLMResult[T]:
    """Читает записанный ответ. Режим `replay`.

    Разбирает **по той же схеме**, что настоящий вызов: запись, не
    подходящая схеме, поднимает `SchemaError`, а не возвращается как есть.
    Иначе устаревшая фикстура прошла бы в тестах и упала в работе.
    """
    path = recording_path(purpose, key, root=root)
    if not path.is_file():
        raise RecordingMissingError(
            f"нет записи для {purpose}/{key}: ожидался файл {path}. "
            "Записать: LLM_MODE=record с ключом провайдера."
        )

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = schema.model_validate(raw["value"])
    except ValidationError as error:
        raise SchemaError(
            f"запись {path} не разбирается по схеме {schema.__name__}: {error}"
        ) from error

    return LLMResult(
        value=value,
        usage=TokenUsage(
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
        ),
        model=str(raw.get("model", "unknown")),
    )
