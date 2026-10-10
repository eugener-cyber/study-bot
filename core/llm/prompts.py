"""Загрузка промптов из файлов. План реализации §1.5, ТЗ §6.1.

Промпты лежат отдельными файлами `core/llm/prompts/<имя>_v<N>.md`, а не
строками в коде. Причина не в красоте: §1.5 требует, чтобы смена версии
промпта делала старые записи `tests/golden/` недействительными, а для этого
версия должна быть **частью текста**, по которому считается ключ записи.
Файл с версией в имени даёт это даром и заодно делает дифф правки промпта
читаемым — промпт в тройных кавычках внутри модуля diff показывает как
перемешанные строки кода и текста.

**Версия подставляется загрузчиком, а не пишется в файле.** Строка `[промпт:
facts_v1]` добавляется первой строкой при рендере. Если бы версию писал
автор промпта, рано или поздно она разошлась бы с именем файла, и ключ записи
перестал бы меняться при смене версии — то есть сломался бы ровно тот
механизм, ради которого версия нужна.

**Файлов промптов в этом пакете нет.** WP-04 — инфраструктура вызова; сами
промпты стадий приходят с WP-05 и далее, каждый со своим пакетом. Загрузчик
без промптов не бесполезен: он задаёт их формат и падает с понятным
сообщением, а не выдумывает текст.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

VERSION_PATTERN = re.compile(r"^(?P<name>[a-z0-9_]+)_v(?P<number>\d+)\.md$")
"""Имя файла промпта. Версия — целое число, а не дата: §1.5 говорит о «версии
промпта», и порядок версий должен быть очевиден из имени."""

VERSION_LINE = "[промпт: {version}]"
"""Первая строка готового промпта.

Модель эту строку видит, и это допустимо: она короткая, не меняет задачу и
помогает при разборе записанного ответа — по первой строке видно, какой
версией он получен.
"""


class PromptMissingError(Exception):
    """Промпта с таким именем нет.

    Сообщение называет каталог и найденные имена: промпт не загружается по
    опечатке в имени так же часто, как по отсутствию файла, и различить эти
    два случая нужно сразу.
    """


class PromptPlaceholdersError(Exception):
    """Набор подстановок не совпал с набором мест в шаблоне.

    Отдельный тип, потому что это самая дорогая из возможных ошибок
    промпта: лишний ключ подстановки означает, что текст материала **не
    попал** в промпт, а модель всё равно ответит — связно и мимо задачи.
    Такой ответ пройдёт проверку схемы и попадёт в факты.
    """


@dataclass(frozen=True)
class Prompt:
    """Загруженный промпт."""

    name: str
    version: str
    """`<имя>_v<N>` — то, что попадает в `llm_calls.prompt_version` (§3)."""

    template: str

    def placeholders(self) -> set[str]:
        """Имена подстановок в шаблоне: `$fragment`, `$section` и подобные."""
        return set(Template(self.template).get_identifiers())

    def render(self, **values: object) -> str:
        """Готовый промпт: строка версии плюс шаблон с подстановками.

        Расхождение набора подстановок — исключение в обе стороны. Нехватку
        поймал бы и `Template.substitute`, а лишний ключ он проигнорировал бы
        молча; именно лишний ключ опаснее, см. `PromptPlaceholdersError`.
        """
        expected = self.placeholders()
        given = set(values)
        if expected != given:
            raise PromptPlaceholdersError(
                f"промпт {self.version}: в шаблоне {sorted(expected)}, "
                f"передано {sorted(given)}; "
                f"не хватает {sorted(expected - given)}, лишние {sorted(given - expected)}"
            )
        body = Template(self.template).substitute(
            {key: str(value) for key, value in values.items()}
        )
        return f"{VERSION_LINE.format(version=self.version)}\n{body}"


def available(directory: Path | None = None) -> dict[str, int]:
    """Имя промпта → наибольший найденный номер версии."""
    root = directory or PROMPT_DIR
    if not root.is_dir():
        return {}

    found: dict[str, int] = {}
    for path in root.iterdir():
        match = VERSION_PATTERN.match(path.name)
        if match is None:
            continue
        name = match.group("name")
        number = int(match.group("number"))
        found[name] = max(found.get(name, 0), number)
    return found


def load_prompt(name: str, *, version: int | None = None, directory: Path | None = None) -> Prompt:
    """Загружает промпт по имени; по умолчанию — последнюю версию.

    Явная версия нужна, когда промпт меняется, а старые записи ещё должны
    воспроизводиться: тест старой стадии просит свою версию и не зависит от
    того, что к ней добавили позже.
    """
    root = directory or PROMPT_DIR
    versions = available(root)
    if name not in versions:
        raise PromptMissingError(
            f"промпта {name!r} нет в {root}; есть: {sorted(versions) or 'ни одного'}"
        )

    number = versions[name] if version is None else version
    path = root / f"{name}_v{number}.md"
    if not path.is_file():
        raise PromptMissingError(f"промпта {name} версии {number} нет: ожидался файл {path}")

    return Prompt(
        name=name,
        version=f"{name}_v{number}",
        template=path.read_text(encoding="utf-8"),
    )
