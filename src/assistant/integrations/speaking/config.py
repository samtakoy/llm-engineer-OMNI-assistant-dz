"""
Настройки синтеза речи одним объектом.

Значения приходят снаружи: пакет не читает окружение и не знает о проекте,
в который встроен.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class SpeakingConfig:
    """
    Настройки модели синтеза и записи звука в файл.

    Атрибуты:
        model_id: версия модели silero: v5_5_ru, v4_ru, v3_1_ru. Набор голосов
            принадлежит версии и меняется вместе с ней.
        language: язык модели двумя буквами.
        device: устройство вычислений: auto, cpu, cuda. Auto берёт видеокарту,
            если она видна torch.
        sample_rate: частота дискретизации синтеза: 8000, 24000, 48000.
        put_accent: расставлять ударения. Работает только при озвучке чистым
            текстом: в режиме ssml silero этот флаг не принимает.
        put_yo: восстанавливать букву ё. Ограничение то же, что у ударений.
        hub_directory: куда torch.hub складывает копию репозитория silero и
            файл весов модели; None - каталог torch по умолчанию.
    """

    model_id: str
    language: str
    device: str
    sample_rate: int
    put_accent: bool
    put_yo: bool
    hub_directory: Path | None
