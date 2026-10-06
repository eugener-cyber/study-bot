"""Сообщение прогресса обработки. ТЗ §4.5.

«Одно сообщение, редактируемое `edit_message_text` не чаще раза в 5 секунд:
`⏳ Распознаю аудио… 4:12 / 58:30`».

Ограничение частоты — не оптимизация. `edit_message_text` чаще упирается в
лимиты Telegram, и ответ приходит не отказом, а `TelegramRetryAfter`, то есть
задержкой всего обработчика. При стадии, рапортующей о каждой странице
трёхсотстраничного скана, это означает обработку, которая ждёт Telegram
дольше, чем работает.

Троттлер исходящих сообщений §17.3 — другой механизм и в этот пакет не входит.
Когда он появится, локальное ограничение здесь снимается: отметка оставлена
нарочно, чтобы два троттлера не остались рядом.
"""

from __future__ import annotations

import time

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from core.logging import get_logger

log = get_logger(__name__)

MIN_INTERVAL_SEC = 5.0
"""§4.5 называет интервал буквально: не чаще раза в 5 секунд."""

NOT_MODIFIED = "message is not modified"
"""Фрагмент ответа Telegram при правке на тот же текст.

Та же история, что с кнопкой согласия в WP-01: отдельного класса исключения в
aiogram 3 нет, различение по тексту. Здесь это достижимо штатно — две стадии
подряд могут дать одинаковую строку прогресса.
"""


class ProgressMessage:
    """Одно редактируемое сообщение на весь конвейер.

    Одно, а не новое на каждую стадию: шесть сообщений о прогрессе засоряют
    чат и оставляют пользователя гадать, какое из них актуально. §4.5 требует
    именно одного.
    """

    def __init__(self, message: Message, *, min_interval: float = MIN_INTERVAL_SEC) -> None:
        self._message = message
        self._min_interval = min_interval
        self._last_sent = 0.0
        self._last_text = ""
        self.skipped = 0
        """Сколько правок отброшено частотой. Для тестов и для логов: по нулю
        видно, что ограничение не срабатывало, и наоборот."""

    async def update(self, text: str, *, force: bool = False) -> bool:
        """Правит сообщение, если пора. Возвращает, была ли правка.

        `force` — для последнего состояния: итоговое «готово» или «не
        получилось» обязано дойти независимо от того, когда была прошлая
        правка. Без этого пользователь мог бы остаться с текстом «обрабатываю»
        на завершённом материале.
        """
        if text == self._last_text and not force:
            return False

        now = time.monotonic()
        if not force and now - self._last_sent < self._min_interval:
            self.skipped += 1
            return False

        try:
            await self._message.edit_text(text)
        except TelegramBadRequest as error:
            if NOT_MODIFIED not in str(error):
                raise
            log.info("progress_not_modified")
            return False

        self._last_sent = now
        self._last_text = text
        return True


def stage_text(stage: str, done: int | None = None, total: int | None = None) -> str:
    """Строка прогресса для стадии.

    Формулировки от первого лица и в настоящем времени — как в примере §4.5
    («Распознаю аудио…»). Пользователь читает их как отчёт о том, что
    происходит сейчас, а не как название этапа в системе.
    """
    titles = {
        "probe": "Смотрю, что в файле",
        "extract": "Разбираю материал",
        "sections": "Выделяю разделы",
        "facts": "Собираю факты",
        "notes": "Пишу конспект",
        "questions": "Составляю задания",
        "schedule": "Ставлю в расписание",
    }
    title = titles.get(stage, stage)
    if done is not None and total:
        return f"⏳ {title}… {done} / {total}"
    return f"⏳ {title}…"
