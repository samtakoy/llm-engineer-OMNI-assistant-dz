"""
Настройки речевого слоя.

Пакет не читает окружение и не знает о проекте, в который встроен: все значения
приходят снаружи одним объектом. Так папку можно скопировать в другой проект и
собрать конфиг там, где этому проекту удобно.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class SpeechConfig:
    """
    Настройки распознавания речи, записи с микрофона и кеша расшифровок.

    Атрибуты:
        recognition_model: модель faster-whisper: small, medium, large-v3.
            На русском small путает имена собственные, medium уже держит.
        recognition_device: устройство вычислений: auto, cpu, cuda.
        recognition_compute_type: тип вычислений: int8, float16, float32.
            На процессоре берут int8, на видеокарте float16.
        language: язык записи двумя буквами. Пустая строка - определять по звуку.
        cache_directory: каталог кеша расшифровок; None выключает кеш.
        cache_ttl_days: сколько дней годна расшифровка. Ноль и меньше - годна
            всегда: запись не меняется, устареть расшифровке нечем.
        bypass_cache: читать мимо кеша. Запись при этом продолжается.
        recording_sample_rate: частота дискретизации записи с микрофона.
            Whisper всё равно приводит звук к 16 кГц, выше писать незачем.
        recording_directory: куда складывать записи с микрофона.
    """

    recognition_model: str
    recognition_device: str
    recognition_compute_type: str
    language: str
    cache_directory: Path | None
    cache_ttl_days: int
    bypass_cache: bool
    recording_sample_rate: int
    recording_directory: Path
