# Команды харнесса (План реализации §1.4).
# В WP-01 существует только check — остальным нечего сбрасывать и нечем
# заполнять до появления схемы в WP-02.

PY := .venv/bin/python

.PHONY: check lint type test install

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
