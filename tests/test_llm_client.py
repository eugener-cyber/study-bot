"""Клиент LLM: два правила ретраев, семафор, учёт. ТЗ §6.1, §6.4, §2.1.

Главное место пакета — **разные правила ретраев для разных ошибок**. §6.1:
«429 и 5xx — экспоненциальный backoff, до 4 попыток; ошибка схемы — один
повтор с приклеенным текстом ошибки, затем `failed`». Написать одно правило на
оба случая легко, и на успешном пути разницы не видно вовсе: она видна только
на негодном ответе модели, где четыре попытки сжигают бюджет на четыре
одинаково негодных ответа.

Поэтому ассерты здесь на **число обращений к провайдеру**, а не на «повтор
был»: вторая формулировка прошла бы и при четырёх попытках.

Файл идёт без маркера `db`: `pytest-socket` закрывает даже loopback, а учёт
проверяется подделкой `ListRecorder`. Ретраи и семафор к базе отношения не
имеют, и держать их на базе значило бы замедлять прогон без причины.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from core.llm.base import (
    LLMResult,
    ProviderError,
    ProviderUnavailableError,
    SchemaError,
    TokenUsage,
)
from core.llm.client import SCHEMA_RETRY_HEADER, CallRecord, LLMClient, RetryPolicy
from core.llm.estimation import PRICING_VERSION
from core.llm.recording import LLMMode, recording_key, save
from core.llm.schemas import SectionsAnswer
from tests.llm_fakes import ListRecorder, ScriptedProvider, fake_client
from tests.test_dispatcher import _settings

PROMPT = "Сгруппируй фрагменты:\n[1] Первый\n[2] Второй"


class Answer(BaseModel):
    text: str


def _paid_client(provider: ScriptedProvider, recorder: ListRecorder, **overrides: str) -> LLMClient:
    """Клиент на платном провайдере: иначе все суммы нулевые (§2.1)."""
    return fake_client(
        _settings(LLM_PROVIDER="anthropic", **overrides),
        provider=provider,
        recorder=recorder,
    )


# --- Успешный путь и семь полей §6.4 --------------------------------------


async def test_successful_call_writes_one_row_with_every_field_of_6_4() -> None:
    """§6.4: «для каждого вызова хранятся provider, model, pricing_version,
    токены, cost_currency, estimated_cost, actual_cost». Плюс `latency_ms` (§3).
    """
    provider = ScriptedProvider(name="anthropic", model="claude-opus-5")
    recorder = ListRecorder()

    await _paid_client(provider, recorder).structured(
        PROMPT,
        SectionsAnswer,
        purpose="sections",
        prompt_version="sections_v1",
        user_id=7,
        material_id=9,
    )

    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert row.provider == "anthropic"
    assert row.model == "claude-opus-5"
    assert row.pricing_version == PRICING_VERSION
    assert (row.input_tokens, row.output_tokens) == (120, 40)
    assert row.cost_currency == "USD"
    assert row.estimated_cost > 0
    assert row.actual_cost > 0
    assert row.latency_ms >= 0
    assert row.ok is True
    assert row.error is None
    assert (row.user_id, row.material_id) == (7, 9)
    assert row.purpose == "sections"
    assert row.prompt_version == "sections_v1"


async def test_acceptance_corridor_of_6_4_is_computable() -> None:
    """Критерий §6.4 — `abs(actual − estimate) / estimate` — вычислим.

    Ассерт именно на вычислимость, а не на попадание в 0.4: на подделке
    попадание ничего не значит, а делимость на оценку — значит. Редакция 2
    плана упёрлась ровно здесь: при нулевой оценке выражение невычислимо.
    """
    recorder = ListRecorder()
    await _paid_client(ScriptedProvider(name="anthropic"), recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="sections_v1"
    )

    row = recorder.rows[0]
    assert row.estimated_cost != Decimal("0")
    assert abs(row.actual_cost - row.estimated_cost) / row.estimated_cost >= 0


async def test_manual_provider_keeps_tokens_and_zeroes_cost() -> None:
    """Ручной режим: строка есть, токены есть, суммы нулевые (§2.1).

    Отсюда и решение ADR-0005: коридор §6.4 считается на токенах, потому что
    на стоимости в этом режиме он `0 / 0`.
    """
    recorder = ListRecorder()
    client = fake_client(_settings(), provider=ScriptedProvider(name="manual"), recorder=recorder)

    await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    row = recorder.rows[0]
    assert row.estimated_cost == Decimal("0")
    assert row.actual_cost == Decimal("0")
    assert row.input_tokens > 0


# --- Правило 1: ошибка схемы — ровно один повтор --------------------------


async def test_schema_error_is_retried_exactly_once() -> None:
    provider = ScriptedProvider(failures=[SchemaError("поле statement пустое")])
    recorder = ListRecorder()

    await fake_client(_settings(), provider=provider, recorder=recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert len(provider.prompts) == 2
    assert len(recorder.rows) == 2


async def test_schema_error_text_is_appended_to_the_retry() -> None:
    """Без текста ошибки повтор просит то же самое теми же словами.

    И получает тот же негодный ответ — то есть расходует деньги и не меняет
    исхода.
    """
    provider = ScriptedProvider(failures=[SchemaError("fragment_ids: пустой список")])

    await fake_client(_settings(), provider=provider).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    retry = provider.prompts[1]
    assert SCHEMA_RETRY_HEADER in retry
    assert "fragment_ids: пустой список" in retry
    assert retry.startswith(PROMPT)


async def test_second_schema_error_gives_up_without_a_third_attempt() -> None:
    """§6.1: «затем `failed`». Третьей попытки нет."""
    provider = ScriptedProvider(failures=[SchemaError("раз"), SchemaError("два")])
    recorder = ListRecorder()

    with pytest.raises(SchemaError, match="два"):
        await fake_client(_settings(), provider=provider, recorder=recorder).structured(
            PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
        )

    assert len(provider.prompts) == 2
    assert len(recorder.rows) == 2
    assert [row.ok for row in recorder.rows] == [False, False]


async def test_failed_rows_carry_the_error_text() -> None:
    """§6.4 учитывает и неудачный вызов: платить приходится за все.

    Текст ошибки в строке нужен не для красоты: без него в истории останется
    `ok = false` без причины, и разбирать придётся по логам, которые к тому
    времени провернутся.
    """
    provider = ScriptedProvider(failures=[SchemaError("options: 5 вариантов")])
    recorder = ListRecorder()

    await fake_client(_settings(), provider=provider, recorder=recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert recorder.rows[0].error is not None
    assert "options: 5 вариантов" in recorder.rows[0].error
    assert "SchemaError" in recorder.rows[0].error


# --- Правило 2: 429 и 5xx — backoff до четырёх попыток --------------------


async def test_unavailable_provider_is_retried_up_to_four_attempts() -> None:
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429")] * 4)
    recorder = ListRecorder()

    with pytest.raises(ProviderUnavailableError):
        await fake_client(_settings(), provider=provider, recorder=recorder).structured(
            PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
        )

    assert len(provider.prompts) == 4
    assert len(recorder.rows) == 4


async def test_recovery_before_the_limit_succeeds() -> None:
    """Обратное направление: три отказа и успех на четвёртой попытке.

    Без него прошла бы и реализация, которая всегда исчерпывает лимит.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("503")] * 3)
    recorder = ListRecorder()

    result = await fake_client(_settings(), provider=provider, recorder=recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert result.value.sections
    assert len(provider.prompts) == 4
    assert [row.ok for row in recorder.rows] == [False, False, False, True]


async def test_network_retry_does_not_change_the_prompt() -> None:
    """Сетевой отказ — не повод править запрос.

    Текст ошибки приклеивается только к ошибке схемы: 429 не означает, что
    модель не поняла задачу.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429")])

    await fake_client(_settings(), provider=provider).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert provider.prompts[0] == provider.prompts[1] == PROMPT


async def test_retry_counters_are_independent() -> None:
    """Сетевой сбой не расходует право на повтор по схеме, и наоборот.

    Один общий счётчик дал бы: 429, затем ошибка схемы — и отказ без второй
    попытки, хотя §6.1 обещает по повтору каждому правилу.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429"), SchemaError("схема")])
    recorder = ListRecorder()

    await fake_client(_settings(), provider=provider, recorder=recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert len(provider.prompts) == 3
    assert [row.ok for row in recorder.rows] == [False, False, True]


async def test_other_provider_errors_are_not_retried() -> None:
    """4xx, кроме 429, повтором не лечится: ошибка в самом запросе.

    `ProviderError` без уточнения типа проходит наружу с первой попытки — и
    строка учёта всё равно пишется.
    """
    provider = ScriptedProvider(failures=[ProviderError("400 bad request")])
    recorder = ListRecorder()

    with pytest.raises(ProviderError, match="400"):
        await fake_client(_settings(), provider=provider, recorder=recorder).structured(
            PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
        )

    assert len(provider.prompts) == 1
    assert len(recorder.rows) == 1


async def test_rows_count_matches_provider_attempts_not_client_calls() -> None:
    """Строк столько, сколько было обращений к провайдеру.

    Счёт по вызовам клиента потерял бы повторы, а платить приходится за них
    тоже.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429"), SchemaError("схема")])
    recorder = ListRecorder()
    client = fake_client(_settings(), provider=provider, recorder=recorder)

    await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")
    await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    assert len(provider.prompts) == 4
    assert len(recorder.rows) == len(provider.prompts)


async def test_failure_cost_uses_the_counters_from_the_exception() -> None:
    """Провайдер, принёсший счётчики отказа, учитывается по ним.

    Негодный ответ оплачен так же, как годный: §6.4 «платить приходится за
    все». Если счётчиков нет, за факт принимается оценка входа при нулевом
    выходе — это нижняя граница, и `ok = false` рядом не даёт принять её за
    успешный расход.
    """
    spent = TokenUsage(input_tokens=1000, output_tokens=700)
    provider = ScriptedProvider(
        name="anthropic", failures=[SchemaError("негодный ответ", usage=spent)]
    )
    recorder = ListRecorder()

    await _paid_client(provider, recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    failed = recorder.rows[0]
    assert (failed.input_tokens, failed.output_tokens) == (1000, 700)
    assert failed.actual_cost == Decimal("5") * 1000 / 1_000_000 + Decimal("25") * 700 / 1_000_000


# --- Семафор §6.1 ---------------------------------------------------------


async def test_concurrency_is_limited_and_actually_happens() -> None:
    """Ассерт двусторонний: не больше предела и **ровно** предел.

    Верхняя граница одна прошла бы и при снятом семафоре — это показала
    мутация «семафор снят», которую прежняя редакция теста не поймала:
    подделка провайдера не уступала управление, одновременных вызовов не
    возникало вовсе, и `peak` оставался единицей.

    Нижняя граница — то, что параллелизм вообще есть: при шестнадцати
    разрешённых и тридцати двух задачах максимум одновременных обязан
    достигнуть шестнадцати, иначе предел §6.1 не предел, а последовательное
    исполнение.

    Ассерт на максимум одновременных, а не на суммарное время: второй
    зависит от машины и даёт мерцающий тест.
    """
    settings = _settings()
    provider = ScriptedProvider()
    client = fake_client(settings, provider=provider)

    await asyncio.gather(
        *[
            client.structured(
                f"{PROMPT}\n{index}", SectionsAnswer, purpose="sections", prompt_version="v1"
            )
            for index in range(settings.LLM_CONCURRENCY * 2)
        ]
    )

    assert provider.peak == settings.LLM_CONCURRENCY
    assert len(provider.prompts) == settings.LLM_CONCURRENCY * 2


async def test_lower_limit_is_respected_too() -> None:
    """Предел берётся из настроек, а не зашит.

    Проверка только на умолчании прошла бы и при зашитой шестнадцати.
    """
    settings = _settings(LLM_CONCURRENCY="2")
    provider = ScriptedProvider()
    client = fake_client(settings, provider=provider)

    await asyncio.gather(
        *[
            client.structured(
                f"{PROMPT}\n{index}", SectionsAnswer, purpose="sections", prompt_version="v1"
            )
            for index in range(8)
        ]
    )

    assert provider.peak == 2


# --- Политика backoff -----------------------------------------------------


def test_delays_grow_and_never_invert() -> None:
    """Монотонность гарантирована, а не вероятна.

    Джиттер 0.2 меньше `(factor − 1) / (factor + 1)` при множителе 2, поэтому
    худший случай предыдущей паузы ниже лучшего случая следующей. Ассерт на
    порядок, а не на абсолютные секунды: §6.1 чисел не задаёт, и тест на
    конкретные секунды был бы тестом реализации.
    """
    policy = RetryPolicy()
    longest = [policy.delay(n, rand=lambda: 1.0) for n in range(1, 5)]
    shortest = [policy.delay(n, rand=lambda: 0.0) for n in range(1, 5)]

    assert all(longest[i] < shortest[i + 1] for i in range(3))


def test_jitter_actually_varies_the_delay() -> None:
    """Без этой проверки прошёл бы и джиттер, равный нулю.

    А он нужен: при `LLM_CONCURRENCY = 16` шестнадцать одновременных повторов
    после 429 получили бы 429 снова.
    """
    policy = RetryPolicy()
    assert policy.delay(1, rand=lambda: 0.0) < policy.delay(1, rand=lambda: 1.0)


def test_delay_is_capped() -> None:
    """Предел применяется после джиттера, иначе он не предел."""
    policy = RetryPolicy(base_sec=100.0, max_sec=30.0)
    assert policy.delay(4, rand=lambda: 1.0) == 30.0


def test_policy_comes_from_settings() -> None:
    """Числа §30.3, а не литералы в клиенте."""
    policy = RetryPolicy.from_settings(_settings(LLM_RETRY_ATTEMPTS="7", LLM_RETRY_MAX_SEC="5"))
    assert policy.attempts == 7
    assert policy.max_sec == 5.0


async def test_attempts_limit_is_configurable() -> None:
    """Предел попыток берётся из настроек и на пути ретраев тоже."""
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429")] * 4)
    client = fake_client(_settings(LLM_RETRY_ATTEMPTS="2"), provider=provider, policy=None)

    with pytest.raises(ProviderUnavailableError):
        await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    assert len(provider.prompts) == 2


# --- Режимы ---------------------------------------------------------------


async def test_replay_reads_the_recording_and_skips_the_provider(tmp_path: Path) -> None:
    """Критерий приёмки: `replay` работает без сети.

    Провайдер здесь сетевой (`offline = False`), и обращений к нему быть не
    должно вовсе: именно это означает «тесты не ходят в сеть». Строка в
    `llm_calls` тоже не пишется — воспроизведённый вызов не стоит ничего, а
    §6.4 это учёт расхода, не журнал обращений.
    """
    provider = ScriptedProvider(name="anthropic")
    provider.offline = False
    recorder = ListRecorder()
    client = fake_client(
        _settings(LLM_MODE="replay"),
        provider=provider,
        recorder=recorder,
        recordings_root=tmp_path,
    )

    prompt = "промпт для записи"
    save(
        "sections",
        recording_key(prompt, Answer),
        LLMResult(Answer(text="из записи"), TokenUsage(10, 5), "claude-opus-5"),
        root=tmp_path,
    )

    result = await client.structured(prompt, Answer, purpose="sections", prompt_version="v1")

    assert result.value.text == "из записи"
    assert provider.prompts == []
    assert recorder.rows == []


async def test_replay_without_a_recording_names_the_expected_path(tmp_path: Path) -> None:
    """Молчаливое «пустой ответ» дало бы отладку по странному поведению конвейера."""
    from core.llm.recording import RecordingMissingError

    provider = ScriptedProvider(name="anthropic")
    provider.offline = False
    client = fake_client(_settings(LLM_MODE="replay"), provider=provider, recordings_root=tmp_path)

    with pytest.raises(RecordingMissingError) as error:
        await client.structured("нет записи", Answer, purpose="sections", prompt_version="v1")

    assert str(tmp_path) in str(error.value)


async def test_offline_provider_is_not_intercepted_in_replay() -> None:
    """`manual` в режиме `replay` идёт к провайдеру, а не к записям.

    Иначе подготовленное руками содержание на томе §30.1 игнорировалось бы в
    режиме по умолчанию — продукт не видел бы собственных данных.
    """
    provider = ScriptedProvider(name="manual")
    assert provider.offline is True
    client = fake_client(_settings(LLM_MODE="replay"), provider=provider)

    await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    assert len(provider.prompts) == 1
    assert client.mode is LLMMode.REPLAY


async def test_record_mode_writes_a_file_that_replay_can_read(tmp_path: Path) -> None:
    """`record` пишет запись, и записанное читается обратно.

    Проверяется кругом: файл мог записаться нечитаемым, и ассерт на его
    существование этого не поймал бы.
    """
    from core.llm.recording import load

    provider = ScriptedProvider(name="manual")
    client = fake_client(_settings(LLM_MODE="record"), provider=provider, recordings_root=tmp_path)

    prompt = f"{PROMPT}\nзапись"
    result = await client.structured(
        prompt, SectionsAnswer, purpose="sections", prompt_version="v1"
    )
    restored = load(
        "sections", recording_key(prompt, SectionsAnswer), SectionsAnswer, root=tmp_path
    )

    assert restored.value.sections[0].fragment_ids == result.value.sections[0].fragment_ids
    assert restored.usage.input_tokens == result.usage.input_tokens


async def test_record_mode_still_calls_the_provider_and_accounts_for_it(
    tmp_path: Path,
) -> None:
    """Запись не отменяет ни вызова, ни учёта: деньги на неё тратятся."""
    provider = ScriptedProvider(name="manual")
    recorder = ListRecorder()
    client = fake_client(
        _settings(LLM_MODE="record"),
        provider=provider,
        recorder=recorder,
        recordings_root=tmp_path,
    )

    await client.structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    assert len(provider.prompts) == 1
    assert len(recorder.rows) == 1


# --- Сбой самого учёта ----------------------------------------------------


class BrokenRecorder:
    """Recorder, который всегда падает. Достижимо: перезапуск Postgres."""

    def __init__(self) -> None:
        self.attempts = 0

    async def record(self, record: CallRecord) -> None:
        self.attempts += 1
        raise RuntimeError("база учёта недоступна")


async def test_recording_failure_does_not_cut_the_retries() -> None:
    """Сбой учёта не меняет правил §6.1.

    Нашло ревью PR #27: `_record_failure` вызывался внутри `except
    ProviderError`, и исключение recorder'а уходило наружу **до** решения о
    повторе. Следствие: гарантия «до 4 попыток» молча зависела от доступности
    базы учёта, а наружу шла не та ошибка — «база учёта недоступна» вместо
    429, причём без строки в `llm_calls`.

    Ассерт на число обращений к провайдеру, а не на отсутствие исключения:
    второе прошло бы и при реализации, которая гасит ошибку recorder'а, но
    выходит из цикла.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429")] * 2)
    recorder = BrokenRecorder()

    result = await fake_client(_settings(), provider=provider, recorder=recorder).structured(
        PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
    )

    assert result.value.sections
    assert len(provider.prompts) == 3
    assert recorder.attempts == 3


async def test_recording_failure_does_not_discard_a_paid_answer() -> None:
    """Успешный путь: ответ уже получен и уже оплачен.

    Выбросить его из-за того, что не записалась строка о нём, — худший из
    возможных обменов.
    """
    recorder = BrokenRecorder()
    result = await fake_client(
        _settings(), provider=ScriptedProvider(), recorder=recorder
    ).structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1")

    assert result.value.sections
    assert recorder.attempts == 1


async def test_the_original_provider_error_still_reaches_the_caller() -> None:
    """Исчерпав попытки, наружу уходит ошибка провайдера, а не учёта.

    Диагностика по подменённой ошибке ведёт в соседнюю подсистему: причина
    отказа материала — 429, а не недоступность базы учёта.
    """
    provider = ScriptedProvider(failures=[ProviderUnavailableError("429 от провайдера")] * 4)

    with pytest.raises(ProviderUnavailableError, match="429 от провайдера"):
        await fake_client(_settings(), provider=provider, recorder=BrokenRecorder()).structured(
            PROMPT, SectionsAnswer, purpose="sections", prompt_version="v1"
        )


async def test_lost_row_is_logged_with_its_whole_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Потеря строки учёта не молчаливая: число восстановимо из журнала.

    Это и есть та уступка §6.4, которой оплачена независимость ретраев от
    базы учёта. Без содержимого в логе уступка была бы просто потерей: §6.4
    получил бы провал в истории, а восстановить его было бы нечем.
    """
    provider = ScriptedProvider(name="anthropic", usage=TokenUsage(1234, 567))
    await fake_client(
        _settings(LLM_PROVIDER="anthropic"), provider=provider, recorder=BrokenRecorder()
    ).structured(PROMPT, SectionsAnswer, purpose="sections", prompt_version="sections_v1")

    printed = capsys.readouterr().out
    assert "llm_call_not_recorded" in printed
    assert "1234" in printed
    assert "567" in printed
    assert "sections_v1" in printed
    assert "база учёта недоступна" in printed
