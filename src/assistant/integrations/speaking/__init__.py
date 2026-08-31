"""
Речевой выход: озвучка текста моделью silero.

Состав пакета:
    SpeakingConfig - настройки модели синтеза и записи звука.
    VoiceSettings - имя голоса, темп и высота речи, звуковой эффект.
    Rate, Pitch, EffectStrength - типы значений темпа, высоты и силы эффекта.
    rate_values, pitch_values, strength_values - значения, которые принимает синтез.
    SpeechSynthesizer - список голосов модели и озвучка текста в wav.
    SynthesisOutcome - исход озвучки.
    available_effects, effect_catalog - имена и описания звуковых эффектов.
    sanitize_markup - чистка разметки ssml по белому списку тегов.
    split_into_chunks - резка длинного текста на куски по бюджету символов.
    wrap_speech_parts - разбивка тела ssml на абзацы и предложения.

Настройки приходят объектом SpeakingConfig, сообщения печатаются через print,
из проекта пакет не импортирует ничего. Внешние зависимости - torch и omegaconf,
обе импортируются внутри функций. Исключения наружу не уходят, при неудаче
возвращается исход с причиной.
"""

from .config import SpeakingConfig
from .effects import NO_EFFECT, available_effects, effect_catalog
from .markup import sanitize_markup, split_into_chunks, wrap_speech_parts
from .outcomes import SynthesisOutcome
from .synthesis import SpeechSynthesizer
from .voices import (
    EffectStrength,
    Pitch,
    Rate,
    VoiceSettings,
    pitch_values,
    rate_values,
    strength_values,
)

__all__ = [
    "SpeakingConfig",
    "NO_EFFECT",
    "available_effects",
    "effect_catalog",
    "sanitize_markup",
    "split_into_chunks",
    "wrap_speech_parts",
    "rate_values",
    "pitch_values",
    "strength_values",
    "SynthesisOutcome",
    "SpeechSynthesizer",
    "VoiceSettings",
    "Rate",
    "Pitch",
    "EffectStrength",
]
