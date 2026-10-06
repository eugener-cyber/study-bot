"""Файловые артефакты стадий. ТЗ §4.2, §4.3.

Без базы: здесь проверяется только дисковая часть атомарности. Связка с
транзакцией — в тестах конвейера.
"""

from __future__ import annotations

from pathlib import Path

from core.storage import TMP_DIRNAME, MaterialStorage


def _storage(tmp_path: Path) -> MaterialStorage:
    return MaterialStorage(tmp_path)


def test_tmp_path_matches_spec(tmp_path: Path) -> None:
    """§4.2 называет путь буквально: `.tmp/<material_id>/<stage>/`."""
    storage = _storage(tmp_path)
    assert storage.tmp_dir(7, "extract") == tmp_path / TMP_DIRNAME / "7" / "extract"


def test_prepare_tmp_creates_empty_directory(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    path = storage.prepare_tmp(7, "extract")
    assert path.is_dir()
    assert list(path.iterdir()) == []


def test_prepare_tmp_removes_leftovers(tmp_path: Path) -> None:
    """Остатки прошлой неудачной попытки удаляются.

    §4.3 требует удалять частичные артефакты незавершённой стадии перед
    повторным запуском. Для файлов это и означает пустую директорию: иначе
    повторная попытка увидела бы половину артефактов прошлой и сочла бы
    работу сделанной.
    """
    storage = _storage(tmp_path)
    first = storage.prepare_tmp(7, "extract")
    (first / "половина.png").write_bytes(b"x")

    second = storage.prepare_tmp(7, "extract")
    assert list(second.iterdir()) == []


def test_commit_moves_files_to_permanent(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    source = storage.prepare_tmp(7, "extract")
    (source / "page1.png").write_bytes(b"a")
    (source / "page2.png").write_bytes(b"b")

    moved = storage.commit_tmp(7, "extract")

    assert sorted(p.name for p in moved) == ["page1.png", "page2.png"]
    assert all(p.exists() for p in moved)
    assert not source.exists(), "временная директория не убрана после переноса"


def test_commit_is_a_move_not_a_copy(tmp_path: Path) -> None:
    """Перенос, а не копия.

    Материалы — это видео и сканы, то есть основной объём диска. Копия
    оставила бы два экземпляра каждого файла.
    """
    storage = _storage(tmp_path)
    source = storage.prepare_tmp(7, "extract")
    (source / "big.bin").write_bytes(b"x" * 1024)

    storage.commit_tmp(7, "extract")

    assert not (tmp_path / TMP_DIRNAME / "7" / "extract" / "big.bin").exists()


def test_commit_on_empty_stage_is_noop(tmp_path: Path) -> None:
    """Стадия без файловых артефактов не должна падать.

    Из шести стадий файлы создаёт только `extract`, и то не всегда: у
    текстового PDF фрагменты текстовые. Остальные пять вызовут `commit_tmp`
    на несуществующей директории.
    """
    storage = _storage(tmp_path)
    assert storage.commit_tmp(7, "facts") == []


def test_commit_twice_overwrites_instead_of_failing(tmp_path: Path) -> None:
    """Повторный запуск стадии не падает на уже перенесённом файле.

    §4.3: повторный запуск завершённой стадии — no-op, а незавершённой
    начинается заново. Второй случай даёт файл с тем же именем в постоянном
    хранилище, и `move` без перезаписи упал бы — то есть идемпотентность
    нарушилась бы на файловой части, а не на транзакционной.
    """
    storage = _storage(tmp_path)

    first = storage.prepare_tmp(7, "extract")
    (first / "page1.png").write_bytes("первая попытка".encode())
    storage.commit_tmp(7, "extract")

    second = storage.prepare_tmp(7, "extract")
    (second / "page1.png").write_bytes("вторая попытка".encode())
    moved = storage.commit_tmp(7, "extract")

    assert moved[0].read_bytes() == "вторая попытка".encode()


def test_discard_removes_tmp_without_moving(tmp_path: Path) -> None:
    """Сбой стадии: артефакты удаляются, в постоянное хранилище не попадают.

    Это и есть безопасная сторона несогласованности. База и диск не делят
    транзакцию, и откатить `os.rename` нельзя. Поэтому порядок выбран так,
    что на диске может остаться мусор во временной директории, но в
    постоянном хранилище не окажется файла от стадии, которой нет в
    `completed_stages`.
    """
    storage = _storage(tmp_path)
    source = storage.prepare_tmp(7, "extract")
    (source / "page1.png").write_bytes(b"x")

    storage.discard_tmp(7, "extract")

    assert not source.exists()
    assert not (storage.permanent_dir(7) / "page1.png").exists()


def test_discard_on_missing_directory_is_noop(tmp_path: Path) -> None:
    """Сбой до создания директории — не повод падать в обработчике сбоя."""
    _storage(tmp_path).discard_tmp(7, "extract")


def test_cleanup_removes_all_stages_of_material(tmp_path: Path) -> None:
    """Уборка при старте воркера.

    Контейнер мог упасть между подготовкой артефактов и коммитом, и тогда
    `.tmp` переживёт перезапуск: том `./data/materials` постоянный (§30.1).
    """
    storage = _storage(tmp_path)
    for stage in ("extract", "facts"):
        (storage.prepare_tmp(7, stage) / "f.bin").write_bytes(b"x")
    other = storage.prepare_tmp(8, "extract")
    (other / "f.bin").write_bytes(b"x")

    storage.cleanup_material_tmp(7)

    assert not (tmp_path / TMP_DIRNAME / "7").exists()
    assert other.exists(), "уборка задела чужой материал"


def test_cleanup_keeps_permanent_files(tmp_path: Path) -> None:
    """Уборка временного не трогает уже перенесённое.

    Иначе перезапуск воркера уничтожал бы артефакты успешно завершённых
    стадий, и материал пришлось бы обрабатывать заново целиком.
    """
    storage = _storage(tmp_path)
    source = storage.prepare_tmp(7, "extract")
    (source / "page1.png").write_bytes(b"x")
    storage.commit_tmp(7, "extract")

    storage.cleanup_material_tmp(7)

    assert (storage.permanent_dir(7) / "page1.png").exists()
