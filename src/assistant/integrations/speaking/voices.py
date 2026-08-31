"""
Настройки голоса для синтеза: имя голоса, темп и высота речи, звуковой эффект.

Значения темпа, высоты и силы эффекта объявлены отдельными типами: их знает
разметка ssml, и по ним же собирается схема ответа модели, подбирающей голос.
"""

from typing import Literal, get_args

from pydantic import BaseModel

Rate = Literal["slow", "medium", "fast"]
Pitch = Literal["x-low", "low", "medium", "high", "x-high"]
EffectStrength = Literal["low", "medium", "high"]


class VoiceSettings(BaseModel):
    """
    Настройки синтеза одного куска речи.

    Атрибуты:
        speaker: имя голоса, которое знает загруженная модель синтеза.
        rate: темп речи.
        pitch: высота голоса.
        effect: имя звукового эффекта из реестра эффектов.
        effect_strength: сила звукового эффекта.
    """

    speaker: str
    rate: Rate
    pitch: Pitch
    effect: str
    effect_strength: EffectStrength


def rate_values() -> tuple[str, ...]:
    """
    Отдаёт значения темпа, которые принимает синтез.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(Rate)


def pitch_values() -> tuple[str, ...]:
    """
    Отдаёт значения высоты, которые принимает синтез.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(Pitch)


def strength_values() -> tuple[str, ...]:
    """
    Отдаёт значения силы эффекта, которые принимает синтез.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(EffectStrength)
