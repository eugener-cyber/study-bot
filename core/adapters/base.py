"""Интерфейс источника. ТЗ §5.1.

```python
class SourceAdapter(ABC):
    kind: str
    async def probe(self, path: str) -> SourceMeta: ...
    def extract(self, path: str, ctx: IngestCtx) -> AsyncIterator[Fragment]: ...
```

§5.1: «Новый тип источника = адаптер + вариант `Locator`. Больше ничего не
меняется». Поэтому здесь нет ничего, кроме контракта: ни PDF, ни аудио, ни
зависимостей от них. Реализации появляются по пакетам — PDF в WP-06, аудио в
WP-19, видео в WP-20.

`probe` отделён от `extract` не для удобства, а по требованию §6.3: оценка
стоимости считается **до** `extract`, потому что для сканированного PDF
`extract` — это vision-вызов на каждую страницу, то есть основная статья
расхода. Гейт после него спрашивал бы «обрабатывать?» после того, как деньги
потрачены.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceMeta:
    """Результат `probe` — то, из чего §6.3 строит предварительную оценку.

    Поля необязательны по природе источника: у PDF есть страницы и нет
    длительности, у аудио наоборот. Пустой `SourceMeta` допустим и означает
    «ничего измерить не удалось» — тогда оценка строится по размеру файла, а
    не выдумывается.
    """

    pages: int | None = None
    duration_sec: float | None = None
    text_chars: int | None = None
    vision_pages_ratio: float = 0.0
    """Доля страниц, требующих vision (§5.2). Главный множитель стоимости:
    страница с текстовым слоем стоит на порядок меньше сканированной."""
    extra: dict[str, Any] = field(default_factory=dict)
    """То, что пойдёт в `materials.origin` — `pages`, `duration`, `youtube_id`
    и прочее по §3. Словарь, а не поля, потому что состав зависит от типа
    источника, а §3 объявляет `origin` как JSONB."""


@dataclass(frozen=True)
class IngestCtx:
    """Контекст извлечения, который адаптер получает, но не формирует сам."""

    material_id: int
    tmp_dir: str
    """`.tmp/<material_id>/<stage>/` (§4.2). Адаптер пишет медиа только сюда:
    перенос в постоянное хранилище делается после коммита стадии, иначе на
    диске остаются артефакты формально не выполненной стадии."""


@dataclass(frozen=True)
class ExtractedFragment:
    """Фрагмент до записи в базу.

    Не модель `core.db.models.Fragment` намеренно: адаптер не должен знать ни
    про SQLAlchemy, ни про идентификаторы. §5.1 называет возвращаемый тип
    `Fragment`, но связывать адаптеры со слоем данных значило бы, что новый
    тип источника меняет не только адаптер.
    """

    ord: int
    kind: str
    """text|transcript|slide|image (§3)."""
    locator: dict[str, Any]
    """§3.4. Обязателен: без локатора фрагмент не даёт перехода к источнику."""
    text: str | None = None
    asset_path: str | None = None
    """Путь во временной директории, если фрагмент — изображение."""
    quality: float = 1.0
    """0..1. Множитель `final_confidence` (§7.3), тип хранения `REAL` (§3)."""
    tokens: int | None = None


class SourceAdapter(ABC):
    """Источник материала."""

    kind: str
    """pdf|image|audio|video|web|text — значение `materials.kind` (§3)."""

    @abstractmethod
    async def probe(self, path: str) -> SourceMeta:
        """Измеряет источник, не обрабатывая его.

        Должен быть дешёвым: вызывается до гейта бюджета, то есть до того, как
        пользователь согласился платить. Для PDF это чтение оглавления и
        проверка текстового слоя, не распознавание.
        """

    @abstractmethod
    def extract(self, path: str, ctx: IngestCtx) -> AsyncIterator[ExtractedFragment]:
        """Разбирает источник на фрагменты.

        Асинхронный итератор, а не список: материал может дать тысячи
        фрагментов, и держать их все в памяти незачем. Сигнатура не
        `async def` — функция возвращает итератор, а не корутину.
        """
