"""Подделки слоя LLM для тестов. План реализации §1.5.

Настоящих ответов модели в тестах нет и быть не может: ключа провайдера нет, а
режим `replay` требует записей, которых спайк §34.1 не сделал. Поэтому стадии
конвейера проверяются на **подделке провайдера**, которая строит ответ из
самого промпта: находит в нём идентификаторы фрагментов и отвечает по схеме
назначения.

Почему так, а не фикстурами в `tests/golden`. Ключ записи — хеш от готового
промпта, а промпт содержит идентификаторы фрагментов, которые выдаёт база:
они меняются от прогона к прогону, и фикстура не находилась бы ни разу.
Записи остаются средством для стадий с устойчивым входом, а конвейер
проверяется подделкой.

**Подделка не строится из констант рабочего кода.** Урок WP-01: фикстура,
собранная из тех же значений, что проверяет, проходит при сломанном коде.
Поэтому разбор промпта здесь свой — регулярка по формату `[id] текст`, — и
если `core.ingest.handlers._fragment_listing` сменит формат, подделка
перестанет находить фрагменты и тесты упадут. Это и требуется: формат промпта
— часть договора со стадией.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from core.config import Settings
from core.llm.base import ImageInput, LLMProvider, LLMResult, TokenUsage
from core.llm.client import CallRecord, LLMClient, RetryPolicy
from core.llm.schemas import FactsAnswer, QuestionSet, SectionsAnswer

FRAGMENT_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$", re.MULTILINE)


class ListRecorder:
    """Учёт вызовов в список вместо базы.

    Позволяет проверять §6.4 без Postgres: ретраи и семафор к базе отношения
    не имеют, и тащить её в такие тесты значило бы держать их на `db`-маркере
    без причины.
    """

    def __init__(self) -> None:
        self.rows: list[CallRecord] = []

    async def record(self, record: CallRecord) -> None:
        self.rows.append(record)


class ScriptedProvider(LLMProvider):
    """Отвечает по схеме назначения, разбирая промпт.

    `failures` — очередь исключений: первые вызовы поднимают их по порядку,
    дальше идут успешные ответы. Так проверяются оба правила ретраев §6.1 без
    настоящего провайдера.
    """

    offline = True

    def __init__(
        self,
        *,
        name: str = "manual",
        model: str = "scripted-1",
        failures: Sequence[Exception] = (),
        usage: TokenUsage | None = None,
        answers: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.failures = list(failures)
        self.answers = dict(answers or {})
        """Ответ по назначению вызова вместо построенного из промпта.

        Нужен тестам, которые проверяют реакцию стадии на **конкретный**
        ответ модели: выдуманный идентификатор фрагмента, повтор слота
        `(type, direction)`, факт-вывод. Построенный из промпта ответ всегда
        корректен, и такие пути остались бы непроверенными.
        """
        self.usage = usage or TokenUsage(input_tokens=120, output_tokens=40)
        self.prompts: list[str] = []
        self.purposes: list[str] = []
        self.active = 0
        self.peak = 0

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, purpose: str
    ) -> LLMResult[T]:
        self.prompts.append(prompt)
        self.purposes.append(purpose)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            # Точка передачи управления обязательна. Без неё корутина
            # провайдера выполняется от начала до конца, не уступая циклу
            # событий, и одновременных вызовов не бывает **никогда** —
            # `peak` остаётся единицей, а проверка семафора проходит и при
            # полностью снятом семафоре. Мутация «семафор снят» это и
            # показала: тест не ловил её, потому что проверял отсутствие
            # того, чего и так не могло случиться.
            await asyncio.sleep(0)
            if self.failures:
                raise self.failures.pop(0)
            payload = self.answers.get(purpose) or _answer_for(schema, prompt)
            value = schema.model_validate(payload)
            return LLMResult(value=value, usage=self.usage, model=self.model)
        finally:
            self.active -= 1

    async def vision[T: BaseModel](
        self,
        images: list[ImageInput],
        prompt: str,
        schema: type[T],
        *,
        purpose: str,
    ) -> LLMResult[T]:
        return await self.structured(prompt, schema, purpose=purpose)


def _fragments_from(prompt: str) -> list[tuple[int, str]]:
    return [(int(number), text.strip()) for number, text in FRAGMENT_LINE.findall(prompt)]


def _answer_for(schema: type[BaseModel], prompt: str) -> dict[str, object]:
    """Правдоподобный ответ под схему. Пустым не бывает: §4.6 проверяется на нём."""
    fragments = _fragments_from(prompt)

    if schema is SectionsAnswer:
        return {
            "sections": [{"title": "Материал целиком", "fragment_ids": [i for i, _ in fragments]}]
        }

    if schema is FactsAnswer:
        return {
            "title": "Материал целиком",
            "summary_md": "Подделка провайдера: секция из одного абзаца.",
            "facts": [
                {
                    "statement": text or f"Утверждение из фрагмента {number}",
                    "detail": "Раскрытие подделки.",
                    "kind": "definition",
                    "fragment_ids": [number],
                    "importance": 3,
                    "difficulty": 2,
                    "derived": False,
                    "confidence": 0.9,
                }
                for number, text in fragments
            ],
        }

    if schema is QuestionSet:
        return {
            "questions": [
                {
                    "type": "mcq",
                    "direction": "forward",
                    "difficulty": 2,
                    "payload": {"question": "Что верно?", "options": ["а", "б", "в", "г"]},
                    "answer": {"correct_index": 0},
                },
                {
                    "type": "open",
                    "direction": "forward",
                    "difficulty": 2,
                    "payload": {"question": "Сформулируйте утверждение.", "hint": None},
                    "answer": {
                        "reference": "эталон",
                        "accept": [],
                        "must_include": [],
                    },
                },
            ]
        }

    raise AssertionError(f"подделка не умеет отвечать по схеме {schema.__name__}")


def fake_client(
    settings: Settings,
    *,
    provider: ScriptedProvider | None = None,
    recorder: ListRecorder | None = None,
    policy: RetryPolicy | None = None,
    recordings_root: Path | None = None,
) -> LLMClient:
    """Клиент на подделке, без пауз backoff.

    `sleep` подменяется на мгновенный: проверяется порядок и число попыток, а
    не способность теста ждать семь секунд. Монотонность задержек проверяется
    отдельно, на самой политике.
    """

    async def no_sleep(_: float) -> None:
        return None

    return LLMClient(
        provider or ScriptedProvider(),
        settings,
        recorder or ListRecorder(),
        policy=policy,
        sleep=no_sleep,
        rand=lambda: 0.5,
        recordings_root=recordings_root,
    )
