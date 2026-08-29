"""
Исход синтеза речи: файл со звуком либо причина неудачи.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class SynthesisOutcome:
    """
    Исход озвучки текста.

    Поля:
        path: файл со звуком, None при неудаче.
        error: краткая причина неудачи, пустая строка если всё получилось.
        seconds: длительность звучания.
    """

    path: Path | None
    error: str
    seconds: float
