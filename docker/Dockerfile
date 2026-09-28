# Один образ на бот и ARQ-воркер: по ТЗ §1.4 это два процесса одного образа.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Исходники копируются ДО установки: pip install . собирает сам проект,
# и setuptools ищет пакеты, перечисленные в [tool.setuptools]. Порядок
# «сначала pyproject ради кеша слоёв» здесь не работает — сборка падает
# с «package directory 'bot' does not exist».
COPY pyproject.toml ./
COPY bot ./bot
COPY core ./core

RUN pip install --no-cache-dir .

CMD ["python", "-m", "bot.main"]
