"""Клиент LLM: ретраи, семафор, учёт стоимости. ТЗ §6.1, §6.4, §2.1.

§2.1 требует прямо: «бизнес-логика этих данных не видит — запись в
`llm_calls` выполняет клиент». Поэтому стадия вызывает только клиент и не
знает ни про режимы, ни про ретраи, ни про токены:

```
стадия → LLMClient.structured(prompt, schema, purpose=…, prompt_version=…)
             ├── семафор LLM_CONCURRENCY
             ├── режим record / replay / live
             ├── провайдер (manual | anthropic | yandexgpt)
             ├── ретраи: 429/5xx — backoff до 4 попыток; схема — ровно один повтор
             └── строка в llm_calls на каждое обращение к провайдеру
```

**Два правила ретраев, а не одно.** §6.1: «429 и 5xx — экспоненциальный
backoff, до 4 попыток; ошибка схемы — один повтор с приклеенным текстом
ошибки, затем `failed`». Разница не стилистическая: сетевая ошибка проходит
сама, а ошибка схемы означает, что модель не поняла задачу, и четыре попытки
сожгут бюджет на четыре одинаково негодных ответа. Счётчики попыток поэтому
**независимы**: сетевой сбой не расходует право на повтор по схеме, и наоборот.

**Строка в `llm_calls` пишется на каждое обращение, включая неудачное.** §6.4
хранит расход, а платить приходится и за негодный ответ. Поэтому строк
столько, сколько было обращений к провайдеру, а не сколько вызовов клиента:
при повторе их две.

**Учёт идёт в своей транзакции.** Урок WP-03: `AwaitingUser`, поднятый внутри
`session_scope`, отменял откатом ту самую запись, ради которой вызывался.
Здесь то же самое в более дорогом виде — стадия упала, транзакция откатилась,
и расход на вызовы, которые уже оплачены, исчез из учёта. `DatabaseRecorder`
открывает собственную сессию на каждую строку.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config import Settings
from core.db.models import LlmCall
from core.db.session import session_scope
from core.llm.base import (
    ImageInput,
    LLMProvider,
    LLMResult,
    ProviderError,
    ProviderUnavailableError,
    SchemaError,
    TokenUsage,
)
from core.llm.estimation import PRICING_VERSION, StageCoefficients, prices_for, tokens_in
from core.llm.recording import LLMMode, load, recording_key, recording_path, save
from core.logging import get_logger

log = get_logger(__name__)

SCHEMA_RETRY_HEADER = "Предыдущий ответ не прошёл проверку схемы. Ошибка:"
"""Заголовок приклеиваемого текста ошибки (§6.1).

Приклеивается именно текст ошибки, а не просьба «ответь правильно»: повтор без
указания, что было не так, просит то же самое теми же словами и получает тот
же негодный ответ — то есть расходует деньги и не меняет исхода.
"""


@dataclass(frozen=True)
class CallRecord:
    """Строка `llm_calls` (§3, §6.4). Все семь полей §6.4 плюс `latency_ms`."""

    purpose: str
    prompt_version: str
    provider: str
    model: str
    pricing_version: str
    input_tokens: int
    output_tokens: int
    cost_currency: str
    estimated_cost: Decimal
    actual_cost: Decimal
    latency_ms: int
    ok: bool
    error: str | None = None
    user_id: int | None = None
    material_id: int | None = None


class CallRecorder(Protocol):
    """Куда клиент пишет учёт.

    Протокол, а не прямая запись в базу, по двум причинам. Проверки ретраев и
    семафора не нуждаются в Postgres, а без протокола им пришлось бы либо
    тащить базу, либо не проверять учёт вовсе. И отсутствие умолчания здесь
    намеренно: клиент требует recorder аргументом, так что «забыть учёт»
    нельзя — а именно это и случилось бы с необязательным параметром.
    """

    async def record(self, record: CallRecord) -> None: ...


class DatabaseRecorder:
    """Пишет строку `llm_calls` в своей транзакции."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, record: CallRecord) -> None:
        async with session_scope(self._sessions) as session:
            session.add(
                LlmCall(
                    user_id=record.user_id,
                    material_id=record.material_id,
                    purpose=record.purpose,
                    prompt_version=record.prompt_version,
                    provider=record.provider,
                    model=record.model,
                    pricing_version=record.pricing_version,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost_currency=record.cost_currency,
                    estimated_cost=record.estimated_cost,
                    actual_cost=record.actual_cost,
                    latency_ms=record.latency_ms,
                    ok=record.ok,
                    error=record.error,
                )
            )


@dataclass(frozen=True)
class RetryPolicy:
    """Параметры backoff. §6.1 задаёт «экспоненциальный, до 4 попыток».

    Базу, множитель, джиттер и предел §6.1 не задаёт, и это неназванные
    числа — они приходят из §30.3 через `Settings`, а не живут литералами
    здесь. Умолчания в этом классе существуют только для тестов политики
    самой по себе; рабочий путь всегда идёт через `from_settings`.
    """

    attempts: int = 4
    base_sec: float = 1.0
    factor: float = 2.0
    jitter: float = 0.2
    max_sec: float = 30.0

    @classmethod
    def from_settings(cls, settings: Settings) -> RetryPolicy:
        return cls(
            attempts=settings.LLM_RETRY_ATTEMPTS,
            base_sec=settings.LLM_RETRY_BASE_SEC,
            factor=settings.LLM_RETRY_FACTOR,
            jitter=settings.LLM_RETRY_JITTER,
            max_sec=settings.LLM_RETRY_MAX_SEC,
        )

    def delay(self, failure: int, *, rand: Callable[[], float] = random.random) -> float:
        """Пауза после `failure`-й неудачи (нумерация с единицы).

        Джиттер применяется **до** предела, а не после: иначе пауза могла бы
        превысить `max_sec` на величину джиттера, то есть предел перестал бы
        быть пределом.

        Монотонность задержек при умолчаниях §30.3 гарантирована, а не
        вероятна: джиттер 0.2 меньше, чем `(factor − 1) / (factor + 1)` при
        множителе 2, поэтому верхняя граница предыдущей паузы всегда ниже
        нижней границы следующей. Тест на монотонность поэтому не мерцает.
        """
        base = self.base_sec * self.factor ** max(failure - 1, 0)
        jittered = base * (1.0 + self.jitter * (2.0 * rand() - 1.0))
        return min(jittered, self.max_sec)


class LLMClient:
    """Единственная точка, через которую стадии обращаются к модели."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        recorder: CallRecorder,
        *,
        policy: RetryPolicy | None = None,
        coefficients: StageCoefficients | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
        recordings_root: Path | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._recorder = recorder
        self._policy = policy or RetryPolicy.from_settings(settings)
        self._coefficients = coefficients or StageCoefficients()
        self._sleep = sleep
        self._rand = rand
        self._mode = LLMMode(settings.LLM_MODE)
        self._recordings_root = recordings_root
        self._semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY)
        self._prices = prices_for(provider.name)

    """Каталог записей задаётся параметром `recordings_root`, а не глобальной
    переменной: тест режимов иначе писал бы в `tests/golden` репозитория, то
    есть проверял бы себя, оставляя за собой файлы."""

    @property
    def mode(self) -> LLMMode:
        return self._mode

    @property
    def provider_name(self) -> str:
        """Значение `questions.provider` и `llm_calls.provider` (§3).

        Открыто свойством, потому что стадия обязана записать провайдера в
        `questions` (§3), но не должна получать для этого сам провайдер:
        §2.1 разрешает ей видеть только клиент.
        """
        return self._provider.name

    async def structured[T: BaseModel](
        self,
        prompt: str,
        schema: type[T],
        *,
        purpose: str,
        prompt_version: str,
        user_id: int | None = None,
        material_id: int | None = None,
    ) -> LLMResult[T]:
        async def call(text: str) -> LLMResult[T]:
            return await self._provider.structured(text, schema, purpose=purpose)

        return await self._run(
            call,
            prompt=prompt,
            schema=schema,
            purpose=purpose,
            prompt_version=prompt_version,
            user_id=user_id,
            material_id=material_id,
        )

    async def vision[T: BaseModel](
        self,
        images: Sequence[ImageInput],
        prompt: str,
        schema: type[T],
        *,
        purpose: str,
        prompt_version: str,
        user_id: int | None = None,
        material_id: int | None = None,
    ) -> LLMResult[T]:
        async def call(text: str) -> LLMResult[T]:
            return await self._provider.vision(list(images), text, schema, purpose=purpose)

        return await self._run(
            call,
            prompt=prompt,
            schema=schema,
            purpose=purpose,
            prompt_version=prompt_version,
            user_id=user_id,
            material_id=material_id,
        )

    async def _run[T: BaseModel](
        self,
        call: Callable[[str], Awaitable[LLMResult[T]]],
        *,
        prompt: str,
        schema: type[T],
        purpose: str,
        prompt_version: str,
        user_id: int | None,
        material_id: int | None,
    ) -> LLMResult[T]:
        if self._mode is LLMMode.REPLAY and not self._provider.offline:
            return self._replayed(prompt, schema, purpose=purpose)

        current = prompt
        network_failures = 0
        schema_failures = 0

        while True:
            estimate = self._estimate(current, purpose)
            started = perf_counter()
            try:
                async with self._semaphore:
                    result = await call(current)
            except ProviderError as error:
                latency = _ms(started)
                await self._record_failure(
                    error,
                    estimate=estimate,
                    purpose=purpose,
                    prompt_version=prompt_version,
                    latency_ms=latency,
                    user_id=user_id,
                    material_id=material_id,
                )

                if isinstance(error, ProviderUnavailableError):
                    network_failures += 1
                    if network_failures >= self._policy.attempts:
                        log.warning(
                            "llm_unavailable_giving_up",
                            purpose=purpose,
                            attempts=network_failures,
                        )
                        raise
                    pause = self._policy.delay(network_failures, rand=self._rand)
                    log.info(
                        "llm_retry_after_backoff",
                        purpose=purpose,
                        failure=network_failures,
                        pause_sec=round(pause, 3),
                    )
                    await self._sleep(pause)
                    continue

                if isinstance(error, SchemaError):
                    if schema_failures:
                        # Второй отказ по схеме — повторов больше нет (§6.1).
                        log.warning("llm_schema_failed_twice", purpose=purpose)
                        raise
                    schema_failures = 1
                    current = f"{prompt}\n\n{SCHEMA_RETRY_HEADER}\n{error}"
                    log.info("llm_retry_with_schema_error", purpose=purpose)
                    continue

                raise

            latency = _ms(started)
            await self._record_success(
                result,
                estimate=estimate,
                purpose=purpose,
                prompt_version=prompt_version,
                latency_ms=latency,
                user_id=user_id,
                material_id=material_id,
            )
            if self._mode is LLMMode.RECORD:
                save(
                    purpose,
                    recording_key(current, schema),
                    result,
                    root=self._recordings_root,
                )
            return result

    # ------------------------------------------------------------- режимы

    def _replayed[T: BaseModel](
        self, prompt: str, schema: type[T], *, purpose: str
    ) -> LLMResult[T]:
        """Ответ из `tests/golden` вместо вызова. Режим `replay` (§1.5 Плана).

        Строка в `llm_calls` не пишется, и это решение: §6.4 — учёт расхода,
        а воспроизведённый вызов не стоит ничего. Писать его значило бы
        насчитывать деньги за то, что не происходило, и прогон тестов
        попадал бы в историю расходов наравне с обработкой материалов.
        """
        key = recording_key(prompt, schema)
        result = load(purpose, key, schema, root=self._recordings_root)
        log.info(
            "llm_replayed",
            purpose=purpose,
            key=key,
            path=str(recording_path(purpose, key, root=self._recordings_root)),
        )
        return result

    # --------------------------------------------------------------- учёт

    def _estimate(self, prompt: str, purpose: str) -> TokenUsage:
        """Оценка расхода до вызова (§6.4, `estimated_cost`).

        Ожидаемый выход берётся долей от входа по той же таблице
        `StageCoefficients`, что и оценка бюджета материала (§6.3). Одна
        таблица на оба расчёта обязательна: два механизма разошлись бы, и
        коридор §6.4 показывал бы расхождение там, где его нет.
        """
        input_tokens = tokens_in(prompt)
        output_tokens = int(input_tokens * self._coefficients.output_ratio_for(purpose))
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, estimated=True)

    async def _record_success(
        self,
        result: LLMResult[BaseModel],
        *,
        estimate: TokenUsage,
        purpose: str,
        prompt_version: str,
        latency_ms: int,
        user_id: int | None,
        material_id: int | None,
    ) -> None:
        await self._recorder.record(
            CallRecord(
                purpose=purpose,
                prompt_version=prompt_version,
                provider=self._provider.name,
                model=result.model,
                pricing_version=PRICING_VERSION,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_currency=self._settings.COST_CURRENCY,
                estimated_cost=self._cost(estimate),
                actual_cost=self._cost(result.usage),
                latency_ms=latency_ms,
                ok=True,
                user_id=user_id,
                material_id=material_id,
            )
        )

    async def _record_failure(
        self,
        error: ProviderError,
        *,
        estimate: TokenUsage,
        purpose: str,
        prompt_version: str,
        latency_ms: int,
        user_id: int | None,
        material_id: int | None,
    ) -> None:
        """Строка отказа. §6.4 учитывает и её — платить приходится за все.

        Фактические счётчики берутся из исключения, если провайдер их принёс
        (`ProviderError.usage`). Если нет — за факт принимается оценка входа
        при нулевом выходе: вход провайдер прочитал заведомо, а выход на
        отказе либо не состоялся, либо негоден. Число помечено не как факт
        вызова, а как то, что известно, и `ok = false` рядом не даёт принять
        его за успешный расход.
        """
        usage = error.usage or TokenUsage(
            input_tokens=estimate.input_tokens, output_tokens=0, estimated=True
        )
        await self._recorder.record(
            CallRecord(
                purpose=purpose,
                prompt_version=prompt_version,
                provider=self._provider.name,
                model=self._settings.LLM_MODEL or self._provider.name,
                pricing_version=PRICING_VERSION,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_currency=self._settings.COST_CURRENCY,
                estimated_cost=self._cost(estimate),
                actual_cost=self._cost(usage),
                latency_ms=latency_ms,
                ok=False,
                error=f"{type(error).__name__}: {error}"[:2000],
                user_id=user_id,
                material_id=material_id,
            )
        )

    def _cost(self, usage: TokenUsage) -> Decimal:
        return self._prices.cost(usage.input_tokens, usage.output_tokens)


def _ms(started: float) -> int:
    """Задержка вызова в миллисекундах (§3, `latency_ms`; §34.1)."""
    return max(int((perf_counter() - started) * 1000), 0)
