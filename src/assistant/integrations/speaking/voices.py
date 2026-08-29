"""
Схема настроек голоса: имя голоса, темп и высота речи, звуковой эффект.
"""

from typing import Literal

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
