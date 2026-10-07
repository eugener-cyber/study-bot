# Один образ на бот и ARQ-воркер: по ТЗ §1.4 это два процесса одного образа.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Исходники копируются ДО установки: pip install . собирает сам проект,
# и setuptools ищет пакеты, перечисленные в [tool.setuptools]. Порядок
# «сначала pyproject ради кеша слоёв» здесь не работает — сборка падает
# с «package directory 'bot' does not exist».
COPY pyproject.toml alembic.ini worker.py ./
COPY bot ./bot
COPY core ./core
COPY alembic ./alembic
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN pip install --no-cache-dir . && chmod +x /usr/local/bin/entrypoint.sh

# Точка входа запускает ДВА процесса — бот и ARQ-воркер (§1.4 ТЗ: «бот и
# ARQ-воркер как два процесса одного образа»), и падает, если упал любой из
# них. Пятый сервис compose дал бы пятый контейнер, а §30.1 называет четыре.
CMD ["/usr/local/bin/entrypoint.sh"]
