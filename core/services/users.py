"""Пользователи и согласие на обработку данных.

ТЗ §35: доступ к БД только через `core/services`. Хендлер разбирает апдейт,
вызывает отсюда и рендерит ответ — SQL в `bot/` не появляется.

ТЗ §31: «Согласие на обработку данных фиксируется при первом запуске
(`users.consent_at`)». Перенесено из WP-01 как CR-B, закрывается здесь.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import User
from core.logging import get_logger

log = get_logger(__name__)


async def accept_consent(session: AsyncSession, tg_id: int) -> User:
    """Фиксирует согласие пользователя и возвращает его строку.

    Строка создаётся **в момент согласия**, а не при `/start`. Telegram id —
    персональные данные (§30.2), и их хранение и есть то, на что человек даёт
    согласие; создавать запись до нажатия кнопки значило бы начать обработку
    раньше разрешения.

    Повторное согласие не перезатирает первое: `consent_at` — запись о том,
    когда человек согласился, и сдвигать её назад нечем, а вперёд — неверно.
    За это отвечает `COALESCE` в `ON CONFLICT`.

    Один оператор вместо «прочитать, проверить, вставить»: два одновременных
    нажатия кнопки дают гонку, в которой вторая вставка падает на уникальном
    индексе `users.tg_id`. Проверка в коде её не закрывает — закрывает БД.
    """
    statement = (
        insert(User)
        .values(tg_id=tg_id, created_at=func.now(), consent_at=func.now())
        .on_conflict_do_update(
            index_elements=[User.tg_id],
            # Ссылка на существующую строку, не на предлагаемую: `excluded`
            # дал бы новое время и затёр исходное согласие.
            set_={"consent_at": func.coalesce(User.__table__.c.consent_at, func.now())},
        )
        .returning(User.id)
    )
    user_id = (await session.execute(statement)).scalar_one()

    user = (await session.execute(select(User).where(User.id == user_id))).scalar_one()
    log.info("consent_recorded", user_id=user.id, tg_id=tg_id)
    return user


async def find_by_tg_id(session: AsyncSession, tg_id: int) -> User | None:
    """Пользователь по Telegram id, или `None`.

    Нужен хендлерам, которым согласие уже дано: читать `users` напрямую они
    по §35 не вправе.
    """
    return (await session.execute(select(User).where(User.tg_id == tg_id))).scalar_one_or_none()
