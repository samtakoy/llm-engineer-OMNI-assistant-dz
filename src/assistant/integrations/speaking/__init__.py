"""
Речевой выход: озвучка текста моделью silero.

Состав пакета:
    SpeakingConfig - настройки модели синтеза и записи звука.
    VoiceSettings - имя голоса, темп и высота речи, звуковой эффект.
    SpeechSynthesizer - список голосов модели и озвучка текста в wav.
    SynthesisOutcome - исход озвучки.
    available_effects, effect_catalog - имена и описания звуковых эффектов.

Настройки приходят объектом SpeakingConfig, сообщения печатаются через print,
из проекта пакет не импортирует ничего. Внешние зависимости - torch и omegaconf,
обе импортируются внутри функций. Исключения наружу не уходят, при неудаче
возвращается исход с причиной.
"""

from .config import SpeakingConfig
from .effects import NO_EFFECT, available_effects, effect_catalog
from .outcomes import SynthesisOutcome
from .synthesis import SpeechSynthesizer
from .voices import VoiceSettings

__all__ = [
    "SpeakingConfig",
    "NO_EFFECT",
    "available_effects",
    "effect_catalog",
    "SynthesisOutcome",
    "SpeechSynthesizer",
    "VoiceSettings",
]
