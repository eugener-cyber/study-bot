# Команды харнесса (План реализации §1.4). Всего там восемь целей; существуют
# check (WP-01), migrate/downgrade/reset (WP-02), seed (WP-03), session и
# answer (WP-05), dump-material (WP-06). Остальным нечем работать: evals
# относится к WP-07, cite к WP-11.

PY := .venv/bin/python

.PHONY: check lint type test install migrate downgrade reset seed session answer dump-material

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

# Прогон материала через конвейер целиком (§1.4). Содержательные стадии на
# заглушках до WP-04 — это и есть критерий приёмки WP-03.
#   make seed FILE=путь/к/файлу.md
seed:
	@test -n "$(FILE)" || (echo "укажите FILE=путь/к/файлу"; exit 1)
	$(PY) -m core.harness seed "$(FILE)"

# Сеанс повторения §16: собрать (или продолжить) и показать задание.
#   make session            — владелец берётся первым из ALLOWED_USER_IDS
#   make session TG_ID=123  — конкретный пользователь
session:
	$(PY) -m core.harness session $(if $(TG_ID),--tg-id $(TG_ID),)

# Ответ на текущее задание §21.1 и разбор §22.
#   make answer OPTION=1    — номер варианта, считая с единицы
#   make answer IDK=1       — «не знаю» (§20): отдельный исход, а не вариант
answer:
	$(PY) -m core.harness answer $(if $(OPTION),--option $(OPTION),) $(if $(IDK),--idk,) $(if $(TG_ID),--tg-id $(TG_ID),)

# Фрагменты материала читаемо (§1.4): страница, заголовок, качество, токены.
#   make dump-material ID=3
dump-material:
	@test -n "$(ID)" || (echo "укажите ID=<material_id>"; exit 1)
	$(PY) -m core.harness dump-material $(ID)
