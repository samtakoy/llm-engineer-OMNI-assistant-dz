"""
Речевой слой: запись вопроса с микрофона и распознавание речи.

Пакет не импортирует ничего из проекта, не читает окружение и не пишет в
журнал: все настройки приходят объектом SpeechConfig, а сообщения печатаются
через print. Перенос в другой проект - копирование папки вместе с соседним
модулем filecache.py, на котором держится кеш расшифровок.

Внешние зависимости: faster-whisper для распознавания, sounddevice и numpy для
записи. Обе группы импортируются внутри функций: без микрофона распознавание
работает, а без faster-whisper работает запись.

Ни одна публичная функция не выбрасывает исключений наружу - при неудаче
возвращается исход с краткой причиной.

Пример использования описан в README.md рядом с этим файлом.
"""

from .config import SpeechConfig
from .outcomes import RecordingOutcome, TranscriptOutcome
from .recognition import SpeechRecognizer
from .recording import SILENCE_LEVEL, record

__all__ = [
    "SpeechConfig",
    "TranscriptOutcome",
    "RecordingOutcome",
    "SpeechRecognizer",
    "record",
    "SILENCE_LEVEL",
]
