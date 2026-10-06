"""Состав развёртывания сверяется с §30.1 и §1.4.

Тот же приём, что в `test_thresholds.py` и `test_models.py`: спецификация
читается как источник, а не переписывается в тест. Здесь это особенно уместно
— в WP-02 расхождение compose с §30.1 на один том пришлось выносить отдельным
change request (CR-G), и без проверки оно обнаружилось только на ревью.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "ТЗ.md"
COMPOSE = ROOT / "docker-compose.yml"
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = ROOT / "docker" / "Dockerfile"


def _spec_section(number: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    body = text.split(f"## {number}")[1]
    return body.split("\n## ")[0]


def _spec_containers() -> set[str]:
    """Контейнеры, перечисленные в §30.1."""
    line = next(
        line for line in _spec_section("30.1").splitlines() if line.startswith("Контейнеры:")
    )
    return set(re.findall(r"`([a-z-]+)`", line))


def _spec_volumes() -> set[str]:
    line = next(line for line in _spec_section("30.1").splitlines() if line.startswith("Тома:"))
    return set(re.findall(r"`\./data/([a-z-]+)`", line))


def _compose_services() -> set[str]:
    """Сервисы compose — по отступу, без зависимости от парсера YAML.

    Двух пробелов достаточно: вложенные ключи имеют больший отступ, а
    комментарии отбрасываются. Полный разбор YAML потребовал бы зависимости
    ради одной строки.
    """
    names: set[str] = set()
    for raw in COMPOSE.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"  ([a-z][a-z0-9-]*):", raw)
        if match:
            names.add(match.group(1))
    return names


def _compose_volumes() -> set[str]:
    return set(re.findall(r"- \./data/([a-z-]+):", COMPOSE.read_text(encoding="utf-8")))


def test_spec_section_is_readable() -> None:
    """Если разбор §30.1 сломался, остальные проверки стали бы зелёными зря."""
    assert _spec_containers()
    assert _spec_volumes()


def test_containers_match_spec() -> None:
    """Состав контейнеров совпадает с §30.1 в обе стороны.

    Обе стороны существенны. Лишний сервиcompose — это пятый контейнер, то
    есть расхождение со спецификацией; отсутствующий — неработающая система.
    """
    spec = _spec_containers()
    compose = _compose_services()
    assert spec - compose == set(), f"в §30.1 есть, в compose нет: {sorted(spec - compose)}"
    assert compose - spec == set(), f"в compose есть, в §30.1 нет: {sorted(compose - spec)}"


def test_volumes_match_spec() -> None:
    """Состав томов совпадает с §30.1.

    Расхождение на один том в WP-02 стоило отдельного change request (CR-G).
    Проверка стоит здесь, чтобы следующее обнаружилось прогоном, а не ревью.
    """
    spec = _spec_volumes()
    compose = _compose_volumes()
    assert spec - compose == set(), f"в §30.1 есть, в compose нет: {sorted(spec - compose)}"
    assert compose - spec == set(), f"в compose есть, в §30.1 нет: {sorted(compose - spec)}"


def test_worker_runs_in_the_same_container_as_the_bot() -> None:
    """§1.4: «бот и ARQ-воркер как два процесса одного образа».

    Проверяется, что точка входа запускает оба. Отдельный сервис compose для
    воркера дал бы пятый контейнер и нарушил §30.1 — поэтому процессов два, а
    контейнер один.
    """
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "arq worker.WorkerSettings" in script, "воркер не запускается"
    assert "bot.main" in script, "бот не запускается"


def test_entrypoint_dies_if_either_process_dies() -> None:
    """Контейнер падает, если упал любой из двух процессов.

    Без этого контейнер оставался бы «живым» с одним работающим процессом:
    бот отвечал бы на сообщения, а материалы не обрабатывались — и compose не
    перезапустил бы ничего, потому что контейнер формально работает. Это
    худший вид отказа: система выглядит рабочей.
    """
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "wait -n" in script, "нет ожидания первого завершившегося процесса"


def test_dockerfile_uses_the_entrypoint() -> None:
    """Образ запускает скрипт, а не один процесс напрямую."""
    assert "entrypoint.sh" in DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_ships_migrations() -> None:
    """Миграции попадают в образ.

    Без них `alembic upgrade head` в контейнере невозможен, а схема
    существует только в миграциях — правило из `CLAUDE.md`, проверяемое
    `test_models_match_migrations`.
    """
    body = DOCKERFILE.read_text(encoding="utf-8")
    assert "alembic.ini" in body
    assert "COPY alembic" in body
