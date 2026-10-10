"""Загрузчик промптов. ТЗ §6.1, План реализации §1.5.

Главное свойство, которое здесь проверяется, — **версия попадает в текст**.
На нём держится §1.5: «смена версии промпта автоматически делает старые записи
недействительными, и это видно по падению тестов, а не по странному
поведению». Ключ записи считается от готового промпта, поэтому версия, не
попавшая в текст, означала бы, что записи не устаревают никогда.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.llm.prompts import (
    PROMPT_DIR,
    PromptMissingError,
    PromptPlaceholdersError,
    available,
    load_prompt,
)
from core.llm.recording import recording_key
from core.llm.schemas import FactsAnswer

SHIPPED = {
    "sections": ["fragments"],
    "facts": ["fragments", "title"],
    "questions": ["detail", "fragments", "statement"],
}
"""Промпты пакета и их подстановки — литералами.

Список записан руками, а не получен из `available()`: иначе тест подтверждал
бы содержимое каталога самим содержимым каталога и прошёл бы при удалённом
промпте. Правка промпта, меняющая набор подстановок, обязана править и эту
строку — и тогда расхождение с вызывающей стадией видно в диффе.
"""


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    (tmp_path / "facts_v1.md").write_text("Первая версия: $fragment", encoding="utf-8")
    (tmp_path / "facts_v2.md").write_text(
        "Вторая версия: $fragment\nОтвет {JSON}", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("не промпт", encoding="utf-8")
    return tmp_path


def test_latest_version_is_loaded_by_default(prompts: Path) -> None:
    assert available(prompts) == {"facts": 2}
    assert load_prompt("facts", directory=prompts).version == "facts_v2"


def test_explicit_version_can_be_requested(prompts: Path) -> None:
    """Старая версия доступна по номеру.

    Нужна, когда промпт сменился, а записи старой версии ещё должны
    воспроизводиться: тест стадии просит свою версию и не зависит от того, что
    к ней добавили позже.
    """
    assert load_prompt("facts", version=1, directory=prompts).version == "facts_v1"


def test_rendered_prompt_starts_with_its_version(prompts: Path) -> None:
    rendered = load_prompt("facts", directory=prompts).render(fragment="текст")
    assert rendered.splitlines()[0] == "[промпт: facts_v2]"


def test_version_change_invalidates_the_recording_key(prompts: Path) -> None:
    """§1.5: смена версии делает старые записи недействительными.

    Проверяется через ключ записи, а не через наличие строки в тексте:
    совпадение ключей означало бы, что запись v1 подходит промпту v2, то есть
    механизм не работает, как бы ни выглядел текст.
    """
    first = load_prompt("facts", version=1, directory=prompts).render(fragment="текст")
    second = load_prompt("facts", version=2, directory=prompts).render(fragment="текст")

    assert recording_key(first, FactsAnswer) != recording_key(second, FactsAnswer)


def test_same_version_and_input_give_the_same_key(prompts: Path) -> None:
    """Обратное направление: ключ устойчив.

    Без этой проверки прошёл бы и ключ, зависящий от времени или адреса
    объекта: записи не находились бы вовсе, а первый тест остался бы зелёным.
    """
    prompt = load_prompt("facts", directory=prompts)
    assert recording_key(prompt.render(fragment="A"), FactsAnswer) == recording_key(
        prompt.render(fragment="A"), FactsAnswer
    )
    assert recording_key(prompt.render(fragment="A"), FactsAnswer) != recording_key(
        prompt.render(fragment="B"), FactsAnswer
    )


def test_braces_in_template_survive_rendering(prompts: Path) -> None:
    """Фигурные скобки не считаются подстановкой.

    Промпты содержат примеры JSON, и `str.format` упал бы на каждом из них —
    поэтому подстановка через `string.Template` с `$name`.
    """
    assert "{JSON}" in load_prompt("facts", directory=prompts).render(fragment="x")


def test_extra_placeholder_is_an_error(prompts: Path) -> None:
    """Лишний ключ подстановки — самая дорогая ошибка промпта.

    Он означает, что текст материала **не попал** в промпт, а модель всё
    равно ответит — связно и мимо задачи. Такой ответ пройдёт проверку схемы
    и попадёт в факты. `Template.substitute` лишний ключ игнорирует молча,
    поэтому проверка своя.
    """
    with pytest.raises(PromptPlaceholdersError, match="лишние"):
        load_prompt("facts", directory=prompts).render(fragment="текст", section="лишнее")


def test_missing_placeholder_is_an_error(prompts: Path) -> None:
    with pytest.raises(PromptPlaceholdersError, match="не хватает"):
        load_prompt("facts", directory=prompts).render()


def test_unknown_prompt_names_the_directory_and_what_exists(prompts: Path) -> None:
    with pytest.raises(PromptMissingError, match="facts"):
        load_prompt("notes", directory=prompts)


def test_missing_version_is_reported_separately(prompts: Path) -> None:
    with pytest.raises(PromptMissingError, match="версии 7"):
        load_prompt("facts", version=7, directory=prompts)


def test_nonexistent_directory_gives_no_prompts(tmp_path: Path) -> None:
    assert available(tmp_path / "нет-такого") == {}


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_prompt_loads_with_expected_placeholders(name: str) -> None:
    """Промпты пакета на месте и ждут ровно те подстановки, что подаёт стадия.

    Без этой проверки расхождение имени подстановки нашлось бы только в
    прогоне стадии на живой базе — то есть в самом дорогом месте.
    """
    prompt = load_prompt(name)
    assert prompt.version == f"{name}_v1"
    assert sorted(prompt.placeholders()) == SHIPPED[name]


def test_prompt_directory_belongs_to_the_package() -> None:
    """Каталог промптов — внутри `core/llm`, как требует §6.1."""
    assert PROMPT_DIR.is_dir()
    assert PROMPT_DIR.parts[-2:] == ("llm", "prompts")
