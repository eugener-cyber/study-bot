"""Команда /start и согласие на обработку данных.

ТЗ §31: согласие фиксируется при первом запуске (`users.consent_at`).

Запись в базу появилась в WP-02 вместе с таблицей `users` — перенос был
согласован как CR-B и здесь закрыт. Хендлер её не делает сам: по §35 в
хендлере только разбор апдейта, вызов сервиса и рендер, а запись выполняет
`core/services/users.py`.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.services.users import accept_consent

log = get_logger(__name__)

CONSENT_CALLBACK = "consent:accept"

NOT_MODIFIED = "message is not modified"
"""Фрагмент текста ошибки Telegram при правке сообщения на тот же текст.

Отдельного класса исключения для этого случая в aiogram 3 нет, поэтому
различение идёт по тексту. Если Telegram сменит формулировку, перестанет
срабатывать узкая ветка и пользователь снова увидит безопасное сообщение —
деградация в сторону прежнего поведения, а не в сторону потери согласия.
"""

GREETING = (
    "Здравствуйте.\n\n"
    "Я помогаю готовиться по вашим материалам: строю конспект, составляю задания "
    "и сам возвращаю вас к теме по расписанию, пока она не закрепится.\n\n"
    "Для работы мне нужно хранить загруженные материалы, составленные по ним "
    "задания и историю ваших ответов. Всё это остаётся на вашем сервере."
)

CONSENT_ACCEPTED = (
    "Готово, согласие принято.\n\n" "Приём материалов пока не готов — это ближайшая работа."
)
"""Приглашать прислать материал нельзя, пока бот его не принимает (§19.4):
молчание после приглашения — противоречие внутри одного экрана."""


def consent_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура с единственной кнопкой принятия согласия.

    Возвращает разметку с одной inline-кнопкой, `callback_data` которой равна
    `CONSENT_CALLBACK`. Кнопка одна сознательно: отказ выражается тем, что
    человек её не нажимает, а отдельная кнопка «не согласен» потребовала бы
    экрана, на котором бот объясняет, что без согласия работать не может.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Согласен, начнём", callback_data=CONSENT_CALLBACK)]
        ]
    )


async def handle_start(message: Message) -> None:
    """Приветствие и запрос согласия."""
    log.info("start_command", user_id=message.from_user.id if message.from_user else None)
    await message.answer(GREETING, reply_markup=consent_keyboard())


async def handle_consent(callback: CallbackQuery, session: AsyncSession) -> None:
    """Обрабатывает нажатие кнопки согласия и записывает его.

    Запись идёт до ответа пользователю: если она упадёт, человек увидит
    безопасное сообщение (§31) и не получит «Готово, согласие принято» при
    незаписанном согласии. Обратный порядок давал бы ответ об успехе при
    пустом `consent_at`.
    """
    user_id = callback.from_user.id if callback.from_user else None
    if user_id is not None:
        await accept_consent(session, user_id)
    log.info("consent_accepted", user_id=user_id)

    await callback.answer()
    if not isinstance(callback.message, Message):
        return

    # Повторное нажатие. Два рубежа, потому что одного не хватает: обычно
    # второй callback приходит уже с правленым текстом и отсекается здесь, но
    # при быстром двойном тапе оба приходят до того, как правка применилась,
    # и тогда срабатывает ветка ниже. Без обоих Telegram отвечает
    # «message is not modified», перехватчик ошибок превращает это в
    # «Что-то пошло не так», и пользователь видит сбой на успешном действии.
    if callback.message.text == CONSENT_ACCEPTED:
        log.info("consent_repeat_click", user_id=user_id)
        return

    try:
        await callback.message.edit_text(CONSENT_ACCEPTED)
    except TelegramBadRequest as error:
        if NOT_MODIFIED not in str(error):
            raise
        log.info("consent_race_click", user_id=user_id)


def build_start_router() -> Router:
    """Роутер `/start` и согласия.

    Фабрика, а не модульный `Router`: aiogram запрещает присоединять один
    роутер дважды, и с объектом уровня модуля второй вызов `build_dispatcher`
    в том же процессе падал `RuntimeError: Router is already attached`. Из-за
    этого сборка диспетчера была непроверяемой by construction — ревью PR #3,
    FAIL 4.3. `build_fallback_router` имел эту форму с самого начала.
    """
    router = Router(name="start")
    router.message.register(handle_start, CommandStart())
    router.callback_query.register(handle_consent, F.data == CONSENT_CALLBACK)
    return router
