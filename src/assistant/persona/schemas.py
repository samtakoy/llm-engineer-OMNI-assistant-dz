"""
Схемы структурированного вывода пакета персонажа и режимы его сборки.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from ..integrations.speaking import VoiceSettings


class PersonaMode(Enum):
    """
    Способ превратить облик в указание рассказчику.

        STRUCTURED: поля схемы Persona, разложенные по шаблону.
        FREE: одна свободная фраза.
    """

    STRUCTURED = "structured"
    FREE = "free"


class Persona(BaseModel):
    """Рассказчик, выведенный из облика персонажа на фотографии."""

    name: str = Field(
        description = "Имя персонажа либо описательное название, если имя неизвестно"
    )
    gender: Literal["мужской", "женский", "неопределённый"] = Field(
        description = "Пол персонажа по облику"
    )
    character: str = Field(
        description = "Характер персонажа: нрав, темперамент, отношение к людям"
    )
    address_to_listener: str = Field(
        description = "Само обращение к слушателю, два-четырые слова, без пояснений"
    )
    speech_manner: str = Field(
        description = "Манера речи: темп, длина фраз, интонация, что делает голос узнаваемым"
    )
    favourite_words: list[str] = Field(
        description = "Любимые словечки и присказки персонажа"
    )
    favourite_sounds: list[str] = Field(
        description = "Какие звуки издает в процессе речи (брр, мммб охи, ахи, смешки)"
    )
    attitude_to_subject: str = Field(
        description = "Как персонаж относится к предмету рассказа и что в нём выделяет"
    )


class NarratorVoice(VoiceSettings):
    """Настройки голоса персонажа вместе с полом рассказчика."""

    narrator_gender: Literal["мужской", "женский", "неопределённый"] = Field(
        description = "Пол рассказчика, выведенный из блока про рассказчика"
    )
