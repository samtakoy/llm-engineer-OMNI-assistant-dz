"""
Исходы работы со звуком.

Получилось или нет с краткой причиной
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class TranscriptOutcome:
    """
    Исход распознавания записи.

    Поля:
        text: расшифровка, пустая строка при неудаче и при тишине в записи.
        error: краткая причина неудачи, пустая строка если всё получилось.
        from_cache: расшифровка взята из кеша, а не посчитана заново.
        load_seconds: сколько заняла загрузка модели. Ноль, если модель уже
            была загружена или не понадобилась.
    """

    text: str
    error: str
    from_cache: bool
    load_seconds: float


@dataclass(frozen = True)
class RecordingOutcome:
    """
    Исход записи с микрофона.

    Поля:
        path: файл с записью, None при неудаче.
        error: краткая причина неудачи, пустая строка если всё получилось.
        seconds: длительность записи.
        peak_level: громкость самого громкого отсчёта, от нуля до единицы.
            По ней видно немой микрофон: файл записался, а в нём тишина.
    """

    path: Path | None
    error: str
    seconds: float
    peak_level: float
