"""
Речевой выход: озвучка текста моделью silero.

Состав пакета:
    SpeakingConfig - настройки модели синтеза и записи звука.
    VoiceSettings - имя голоса, темп и высота речи.
    SpeechSynthesizer - список голосов модели и озвучка текста в wav.
    SynthesisOutcome - исход озвучки.

Настройки приходят объектом SpeakingConfig, сообщения печатаются через print,
из проекта пакет не импортирует ничего. Внешние зависимости - torch и omegaconf,
обе импортируются внутри функций. Исключения наружу не уходят, при неудаче
возвращается исход с причиной.
"""

from .config import SpeakingConfig
from .outcomes import SynthesisOutcome
from .synthesis import SpeechSynthesizer
from .voices import VoiceSettings

__all__ = [
    "SpeakingConfig",
    "SynthesisOutcome",
    "SpeechSynthesizer",
    "VoiceSettings",
]
