# Команды харнесса (План реализации §1.4). Всего там восемь целей; существуют
# check (WP-01) и migrate/downgrade/reset (WP-02). Остальным нечем работать:
# seed относится к WP-03, session и answer к WP-05, dump-material к WP-06,
# evals к WP-07, cite к WP-11.

PY := .venv/bin/python

.PHONY: check lint type test install migrate downgrade reset

install:
	python3 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest -q

# Ровно то, что выполняет CI. Один источник: workflow вызывает эту цель,
# а не повторяет цепочку — иначе они разойдутся в первом же пакете.
check: lint type test

migrate:
	$(PY) -m alembic upgrade head

# Откат на одну ревизию. Полный откат — `downgrade base`, но по умолчанию шаг
# один: снести всю схему проще, чем восстановить данные после этого.
downgrade:
	$(PY) -m alembic downgrade -1

# Пересоздание базы: схема поднимается ТОЛЬКО миграциями, никогда create_all.
# Иначе модели и миграции разойдутся, тесты на моделях останутся зелёными, а
# развёрнутая схема будет другой. Правило проверяется в test_migrations.py.
reset:
	$(PY) -m alembic downgrade base
	$(PY) -m alembic upgrade head
