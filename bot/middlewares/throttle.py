"""Ограничение частоты входящих сообщений.

ТЗ §31: не более `THROTTLE_MESSAGES` сообщений от одного пользователя за
`THROTTLE_WINDOW_SEC` секунд. Счётчики в Redis с ключом по `user_id` и временем
жизни, равным окну: переживают рестарт процесса и не требуют отдельной очистки.
При превышении — одно предупреждение за окно, дальше молчание.

Назначение — не защита от атаки: доступ и так закрыт белым списком (§1.3). Это
отсекает случайный цикл, зажатую кнопку и повторную отправку файла, каждая из
которых иначе запускает обработку материала заново и расходует бюджет §6.3.

Троттлер §17.3 на исходящие сообщения — другой механизм, вводится в §17.

Регистрируется outer middleware на `Update`, а не на отдельных обсерверах —
см. `bot/main.py`. Поэтому ответ пользователю идёт через `bot.notify`: у
`Update` метода `answer` нет.

**Единица счёта — апдейт, а не сообщение.** §31 говорит о сообщениях, и после
переноса на уровень `Update` счётчик растёт и от `callback_query`, и от
`inline_query`, и от прочих типов. Это строго строже спецификации, и
назначению §31 такая строгость отвечает: «случайный цикл, зажатая кнопка,
повторная отправка файла» — это ровно callback и прочие апдейты, а не только
текстовые сообщения. Расхождение отмечено здесь сознательно, чтобы следующий
читающий не принял его за недосмотр.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User
from redis.asyncio import Redis

from bot.notify import notify
from core.logging import get_logger

log = get_logger(__name__)

WARNING = "Слишком часто. Подождите немного."


class ThrottleMiddleware(BaseMiddleware):
    """Скользящее окно на счётчике в Redis."""

    def __init__(self, redis: Redis, limit: int, window_sec: int) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_sec

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        count = await self._hit(user.id)

        if count > self._limit:
            # Предупреждаем ровно на первом превышении в окне, дальше молчим:
            # иначе зажатая кнопка превратится в поток предупреждений.
            if count == self._limit + 1:
                log.info("throttled", user_id=user.id, count=count)
                await notify(event, WARNING)
            return None

        return await handler(event, data)

    async def _hit(self, user_id: int) -> int:
        """Увеличивает счётчик и возвращает его значение.

        TTL ставится только при создании ключа (`NX`), иначе окно сдвигалось бы
        с каждым сообщением и никогда не истекало.
        """
        key = f"throttle:{user_id}"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, self._window, nx=True)
        result = await pipe.execute()
        return int(result[0])
