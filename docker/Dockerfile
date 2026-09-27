# Один образ на бот и ARQ-воркер: по ТЗ §1.4 это два процесса одного образа.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY bot ./bot
COPY core ./core

CMD ["python", "-m", "bot.main"]
