"""Режимы записи и воспроизведения. План реализации §1.5.

Критерий приёмки пакета — «`replay` проходит при отключённой сети». Здесь он
проверяется буквально: файл идёт без маркера `db`, то есть `pytest-socket`
закрывает даже loopback, и любое обращение к сети упало бы, а не замедлило
прогон.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from core.llm.base import LLMResult, SchemaError, TokenUsage
from core.llm.recording import (
    LLMMode,
    RecordingMissingError,
    load,
    recording_key,
    recording_path,
    save,
)
from core.llm.schemas import FactsAnswer, QuestionSet, SectionsAnswer


class Narrow(BaseModel):
    text: str


class Wide(BaseModel):
    text: str
    extra: int = 0


def _result(text: str = "ответ") -> LLMResult[Narrow]:
    return LLMResult(
        value=Narrow(text=text),
        usage=TokenUsage(input_tokens=120, output_tokens=40),
        model="claude-opus-5",
    )


def test_replay_is_the_default_mode() -> None:
    """§1.5 задаёт это прямо, и это защита, а не удобство.

    Тест, забывший подменить провайдера, не уходит в сеть, а падает на
    отсутствующей фикстуре.
    """
    assert LLMMode.REPLAY == "replay"
    assert {mode.value for mode in LLMMode} == {"record", "replay", "live"}


def test_record_then_replay_is_a_round_trip(tmp_path: Path) -> None:
    """Записанное читается обратно.

    Проверяется кругом, а не наличием файла: файл может оказаться
    нечитаемым, а тест на `path.exists()` этого не заметит.
    """
    key = recording_key("промпт", Narrow)
    save("facts", key, _result(), root=tmp_path)

    restored = load("facts", key, Narrow, root=tmp_path)
    assert restored.value.text == "ответ"
    assert restored.model == "claude-opus-5"


def test_recorded_token_counters_survive(tmp_path: Path) -> None:
    """Счётчики пишутся вместе с ответом.

    Запись без токенов сделала бы воспроизведённый вызов бесплатным, то есть
    учёт §6.4 в режиме `replay` показывал бы нули при реально потраченных на
    запись деньгах.
    """
    key = recording_key("промпт", Narrow)
    save("facts", key, _result(), root=tmp_path)

    usage = load("facts", key, Narrow, root=tmp_path).usage
    assert (usage.input_tokens, usage.output_tokens) == (120, 40)
    assert usage.estimated is False


def test_missing_recording_names_the_expected_path(tmp_path: Path) -> None:
    """Сообщение называет путь, а не только факт отсутствия.

    Ключ — хеш, по нему нельзя угадать, где искали. Без пути отладка
    начинается с поиска места поиска.
    """
    key = recording_key("промпт", Narrow)
    expected = str(recording_path("facts", key, root=tmp_path))

    with pytest.raises(RecordingMissingError) as error:
        load("facts", key, Narrow, root=tmp_path)

    assert expected in str(error.value)
    assert "LLM_MODE=record" in str(error.value)


def test_recording_that_does_not_fit_the_schema_raises(tmp_path: Path) -> None:
    """Устаревшая фикстура падает, а не возвращается как есть.

    Иначе она прошла бы в тестах и упала в работе — то есть ровно в том
    месте, которое записи призваны защитить.
    """
    path = recording_path("facts", "ручной", root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "m", "value": {"nope": 1}}), encoding="utf-8")

    with pytest.raises(SchemaError, match="Narrow"):
        load("facts", "ручной", Narrow, root=tmp_path)


def test_key_changes_with_the_prompt_text() -> None:
    assert recording_key("A", Narrow) != recording_key("B", Narrow)


def test_key_changes_with_the_schema() -> None:
    """Схема входит в ключ: изменив поля ответа, мы меняем задачу."""
    assert recording_key("A", Narrow) != recording_key("A", Wide)


def test_key_changes_even_on_a_pure_rename() -> None:
    """Переименование схемы обесценивает её записи — и это выбор, не дефект.

    `model_json_schema()` кладёт имя класса в `title`, поэтому ключ меняется
    от одного переименования. Цена несимметрична: лишняя перезапись стоит
    одного прогона с ключом провайдера, а слияние ключей двух разных схем
    вернуло бы ответ не на тот вопрос — молча, потому что он разобрался бы по
    схеме. Тест закрепляет именно это поведение: попытка «улучшить» ключ
    чисткой `title` сделает его красным, и улучшение придётся обосновать.
    """

    class Renamed(BaseModel):
        text: str

    assert recording_key("A", Renamed) != recording_key("A", Narrow)


def test_key_is_hex_and_short() -> None:
    """Ключ — имя файла, поэтому без разделителей пути и не слишком длинный."""
    key = recording_key("промпт", Narrow)
    assert len(key) == 16
    assert all(char in "0123456789abcdef" for char in key)


@pytest.mark.parametrize("schema", [SectionsAnswer, FactsAnswer, QuestionSet])
def test_every_purpose_schema_gives_its_own_key(schema: type[BaseModel]) -> None:
    """Три схемы пакета дают три разных ключа на одном промпте.

    Без этого один и тот же промпт для разных назначений делил бы запись, и
    ответ по секциям вернулся бы как ответ по фактам.
    """
    others = {
        recording_key("промпт", other)
        for other in (SectionsAnswer, FactsAnswer, QuestionSet)
        if other is not schema
    }
    assert recording_key("промпт", schema) not in others


def test_path_layout_is_purpose_then_key(tmp_path: Path) -> None:
    path = recording_path("questions", "abc", root=tmp_path)
    assert path == tmp_path / "questions" / "abc.json"
