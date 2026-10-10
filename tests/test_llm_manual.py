"""Провайдер `manual`: содержание готовит человек. ТЗ §2.1, §34.1.

Это единственная реализация `LLMProvider`, которая сейчас работает, то есть
весь продукт в режиме по умолчанию идёт через неё.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from core.llm.base import ImageInput, SchemaError
from core.llm.manual import (
    MODEL_LABEL,
    ManualContentMissingError,
    ManualProvider,
)
from core.llm.recording import recording_key, recording_path


class Answer(BaseModel):
    statements: list[str]


PROMPT = "Выдели факты: вода кипит при 100 °C"


def _answer_path(root: Path, prompt: str = PROMPT, purpose: str = "facts") -> Path:
    return recording_path(purpose, recording_key(prompt, Answer), root=root)


def _write(root: Path, payload: dict[str, object], prompt: str = PROMPT) -> Path:
    path = _answer_path(root, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_provider_is_offline_and_named_manual() -> None:
    """`offline` существенен: по нему клиент не подменяет вызовы в `replay`.

    Если бы перехватывал, подготовленное руками содержание на томе §30.1 в
    режиме по умолчанию игнорировалось бы — продукт не видел бы собственных
    данных.
    """
    provider = ManualProvider()
    assert provider.name == "manual"
    assert provider.offline is True


async def test_missing_content_writes_a_request_with_prompt_and_schema(tmp_path: Path) -> None:
    """Файл-запрос — рабочий механизм режима, а не отладочный вывод.

    Без него человек видит «нет содержания для ключа 9f3c…» и не знает ни что
    спрашивали, ни в каком виде отвечать: ключ — хеш, и промпт по нему не
    восстановить.
    """
    provider = ManualProvider([tmp_path])

    with pytest.raises(ManualContentMissingError) as error:
        await provider.structured(PROMPT, Answer, purpose="facts")

    request = next(tmp_path.rglob("*.request.md"))
    body = request.read_text(encoding="utf-8")
    assert "вода кипит" in body
    assert "statements" in body
    assert str(_answer_path(tmp_path)) in str(error.value)


async def test_prepared_answer_is_parsed_and_counted_as_estimate(tmp_path: Path) -> None:
    """Токены оценочные, стоимость нулевая — §2.1, решение Архитектора.

    `estimated=True` обязателен: §6.4 сравнивает оценку с фактом, и без
    пометки ручной период выглядел бы идеальным совпадением, то есть коридор
    показывал бы ноль расхождения там, где он ничего не сравнивал.
    """
    _write(tmp_path, {"statements": ["Вода кипит при 100 °C"]})
    provider = ManualProvider([tmp_path])

    result = await provider.structured(PROMPT, Answer, purpose="facts")

    assert result.value.statements == ["Вода кипит при 100 °C"]
    assert result.model == MODEL_LABEL
    assert result.usage.estimated is True
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


async def test_recorded_envelope_is_accepted_but_counters_stay_estimates(tmp_path: Path) -> None:
    """Запись `tests/golden` читается, но её счётчики не присваиваются.

    Вызова не было, и выдать записанные токены за свои значило бы утверждать,
    что модель отвечала сейчас. Настоящие счётчики записи использует режим
    `replay`, где это уместно.
    """
    _write(
        tmp_path,
        {
            "model": "claude-opus-5",
            "input_tokens": 9999,
            "output_tokens": 8888,
            "value": {"statements": ["из записи"]},
        },
    )
    result = await ManualProvider([tmp_path]).structured(PROMPT, Answer, purpose="facts")

    assert result.value.statements == ["из записи"]
    assert result.model == MODEL_LABEL
    assert result.usage.input_tokens != 9999
    assert result.usage.estimated is True


async def test_volume_overrides_the_repository(tmp_path: Path) -> None:
    """Порядок каталогов: сначала том §30.1, потом `tests/golden`.

    Он позволяет перекрыть запись ответом руками, не правя репозиторий.
    Обратный порядок заставлял бы удалять фикстуру, чтобы испытать новый
    ответ.
    """
    volume, golden = tmp_path / "volume", tmp_path / "golden"
    _write(volume, {"statements": ["руками"]})
    _write(golden, {"statements": ["из репозитория"]})

    result = await ManualProvider([volume, golden]).structured(PROMPT, Answer, purpose="facts")
    assert result.value.statements == ["руками"]


async def test_vision_key_depends_on_the_images(tmp_path: Path) -> None:
    """Две страницы с одним промптом — два разных ключа.

    Без участия изображений в ключе вторая страница молча вернула бы
    содержание первой, и материал из сотни сканов получил бы один и тот же
    распознанный текст сто раз.
    """
    provider = ManualProvider([tmp_path])
    keys: list[str] = []
    for page in ("p1.png", "p2.png"):
        with pytest.raises(ManualContentMissingError) as error:
            await provider.vision([ImageInput(path=page)], "Распознай", Answer, purpose="facts")
        keys.append(str(error.value))

    assert keys[0] != keys[1]


async def test_vision_request_lists_the_files_for_the_human(tmp_path: Path) -> None:
    provider = ManualProvider([tmp_path])
    with pytest.raises(ManualContentMissingError):
        await provider.vision(
            [ImageInput(path="/data/page-7.png")], "Распознай", Answer, purpose="facts"
        )

    body = next(tmp_path.rglob("*.request.md")).read_text(encoding="utf-8")
    assert "/data/page-7.png" in body


async def test_answer_not_matching_the_schema_raises_schema_error(tmp_path: Path) -> None:
    """Ответ не по схеме — `SchemaError`, то есть правило ретраев §6.1.

    Один повтор с текстом ошибки имеет смысл и для человека: он увидит, что
    именно не разобралось.
    """
    _write(tmp_path, {"statements": "не список"})

    with pytest.raises(SchemaError, match="Answer"):
        await ManualProvider([tmp_path]).structured(PROMPT, Answer, purpose="facts")


def test_provider_without_roots_is_rejected() -> None:
    """Пустой список каталогов — ошибка на старте, а не на первом вызове."""
    with pytest.raises(ValueError, match="негде"):
        ManualProvider([])


async def test_unwritable_request_still_raises_with_an_explanation(tmp_path: Path) -> None:
    """Если запрос не записался, исключение говорит об этом прямо.

    Иначе человек пойдёт искать файл, которого нет, по пути из сообщения.
    """
    blocked = tmp_path / "занято"
    blocked.write_text("не каталог", encoding="utf-8")

    with pytest.raises(ManualContentMissingError, match="записать не удалось"):
        await ManualProvider([blocked]).structured(PROMPT, Answer, purpose="facts")
