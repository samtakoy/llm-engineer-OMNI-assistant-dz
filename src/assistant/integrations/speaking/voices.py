"""
Схема настроек голоса: имя голоса, темп и высота речи, звуковой эффект.
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field


class VoiceSettings(BaseModel):
    """Настройки синтеза под характер персонажа."""

    speaker: str = Field(
        description = "Имя голоса из доступных в модели синтеза"
    )
    rate: Literal["x-slow", "slow", "medium", "fast", "x-fast"] = Field(
        description = "Темп речи персонажа"
    )
    pitch: Literal["x-low", "low", "medium", "high", "x-high"] = Field(
        description = "Высота голоса персонажа"
    )
    effect: str = Field(
        description = "Имя звукового эффекта из доступных"
    )
    effect_strength: Literal["low", "medium", "high"] = Field(
        description = "Сила звукового эффекта"
    )


def rate_values() -> tuple[str, ...]:
    """
    Отдаёт значения темпа, которые принимает схема.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(VoiceSettings.model_fields["rate"].annotation)


def pitch_values() -> tuple[str, ...]:
    """
    Отдаёт значения высоты, которые принимает схема.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(VoiceSettings.model_fields["pitch"].annotation)


def strength_values() -> tuple[str, ...]:
    """
    Отдаёт значения силы эффекта, которые принимает схема.

    Возвращает:
        Значения в порядке объявления.
    """
    return get_args(VoiceSettings.model_fields["effect_strength"].annotation)

