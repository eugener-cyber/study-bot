"""Файловые артефакты стадий. ТЗ §4.2.

«Файловые артефакты сначала пишутся во временную директорию
`.tmp/<material_id>/<stage>/` и переносятся в постоянное хранилище только
после успешного коммита. Иначе на диске остаются клипы и картинки от стадии,
которая формально не выполнена».

База и диск не делят транзакцию, и это не исправить: нельзя откатить
`os.rename`. Поэтому порядок выбран так, чтобы несогласованность была в
безопасную сторону — на диске может остаться мусор во временной директории,
но в постоянном хранилище не окажется файла от стадии, которой нет в
`completed_stages`. Обратный порядок давал бы ссылку на файл, появившийся до
коммита и переживший откат.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from core.logging import get_logger

log = get_logger(__name__)

TMP_DIRNAME = ".tmp"


class MaterialStorage:
    """Пути материала: временные и постоянные.

    Корень передаётся, а не берётся из настроек: §30.1 называет том
    `./data/materials`, но тесты работают в своей директории, и жёсткая
    привязка к настройкам сделала бы их зависимыми от окружения — та же дыра,
    что с `ALLOWED_USER_IDS` в WP-01, где все тесты шли мимо рабочего пути.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def permanent_dir(self, material_id: int) -> Path:
        return self._root / str(material_id)

    def tmp_dir(self, material_id: int, stage: str) -> Path:
        """`.tmp/<material_id>/<stage>/` — §4.2 буквально."""
        return self._root / TMP_DIRNAME / str(material_id) / stage

    def prepare_tmp(self, material_id: int, stage: str) -> Path:
        """Создаёт пустую временную директорию стадии.

        Пустую: остатки предыдущей неудачной попытки удаляются. §4.3 требует
        удалять частичные артефакты незавершённой стадии перед повторным
        запуском, и для файлов это означает именно это. Иначе повторная
        попытка увидела бы половину артефактов прошлой и сочла бы работу
        сделанной.
        """
        path = self.tmp_dir(material_id, stage)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        return path

    def commit_tmp(self, material_id: int, stage: str) -> list[Path]:
        """Переносит артефакты стадии в постоянное хранилище.

        Вызывается **после** коммита транзакции стадии. Возвращает новые пути.

        Перенос, а не копирование: копия оставила бы два экземпляра каждого
        файла, а материалы — это видео и сканы, то есть основной объём диска.
        """
        source = self.tmp_dir(material_id, stage)
        if not source.exists():
            return []

        target = self.permanent_dir(material_id)
        target.mkdir(parents=True, exist_ok=True)

        moved: list[Path] = []
        for item in sorted(source.iterdir()):
            destination = target / item.name
            if destination.exists():
                # Повторный запуск стадии: файл с тем же именем уже перенесён
                # прошлой попыткой. Перезаписываем — иначе `move` упал бы, и
                # идемпотентность §4.3 нарушилась бы на файловой части.
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
            moved.append(destination)

        shutil.rmtree(source)
        log.info("stage_artifacts_committed", stage=stage, count=len(moved))
        return moved

    def discard_tmp(self, material_id: int, stage: str) -> None:
        """Удаляет временные артефакты стадии, не перенося их.

        Вызывается при сбое стадии. Отдельный метод, а не повторное
        `prepare_tmp`: разница в намерении, и в логе видно, что произошло.
        """
        path = self.tmp_dir(material_id, stage)
        if path.exists():
            shutil.rmtree(path)
            log.info("stage_artifacts_discarded", stage=stage)

    def cleanup_material_tmp(self, material_id: int) -> None:
        """Убирает всю временную директорию материала.

        Вызывается при старте воркера по материалам в статусе `processing`:
        контейнер мог упасть между подготовкой артефактов и коммитом, и тогда
        `.tmp` переживёт перезапуск — том `./data/materials` постоянный (§30.1).

        Отдельной периодической задачи для этого нет сознательно: второй
        механизм уборки без владельца разошёлся бы с первым.
        """
        path = self._root / TMP_DIRNAME / str(material_id)
        if path.exists():
            shutil.rmtree(path)
            log.info("material_tmp_cleaned", material_id=material_id)
