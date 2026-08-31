"""
Схемы структурированного вывода пакета персонажа и режимы его сборки.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

from ..integrations.speaking import EffectStrength, Pitch, Rate, VoiceSettings

Gender = Literal["мужской", "женский", "неопределённый"]

UNKNOWN_GENDER: Gender = "неопределённый"


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
    gender: Gender = Field(
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


class VoiceChoice(BaseModel):
    """Голос, подобранный моделью под манеру рассказчика."""

    voice_name: str = Field(
        description = "Имя голоса из списка доступных, который будет озвучивать рассказчика"
    )
    rate: Rate = Field(
        description = "Темп речи персонажа"
    )
    pitch: Pitch = Field(
        description = "Высота голоса персонажа"
    )
    effect: str = Field(
        description = "Имя звукового эффекта из списка доступных, ровно как оно там записано"
    )
    effect_strength: EffectStrength = Field(
        description = "Сила звукового эффекта"
    )
    narrator_gender: Gender = Field(
        description = "Пол рассказчика, выведенный из блока про рассказчика"
    )

    def to_voice_settings(self) -> VoiceSettings:
        """
        Переводит выбор модели в настройки синтеза.

        Возвращает:
            Настройки синтеза без пола рассказчика: синтезу он не нужен.
        """
        return VoiceSettings(
            speaker = self.voice_name,
            rate = self.rate,
            pitch = self.pitch,
            effect = self.effect,
            effect_strength = self.effect_strength,
        )


@lru_cache(maxsize = 8)
def voice_choice_schema(speakers: tuple[str, ...], effects: tuple[str, ...]) -> type[VoiceChoice]:
    """
    Собирает схему ответа под голоса и эффекты, доступные в этом прогоне.

    Имя голоса и имя эффекта становятся перечислениями.
    Списки приходят снаружи, имена в схеме не хранятся.

        speakers: имена голосов, которые знает модель синтеза.
        effects: имена эффектов реестра.

    Возвращает: Схему ответа модели, подбирающей голос.
    """
    return create_model(
        "VoiceChoiceOfAvailable",
        __base__ = VoiceChoice,
        voice_name = (
            Literal[speakers],
            Field(description = "Имя голоса, который будет озвучивать рассказчика"),
        ),
        effect = (
            Literal[effects],
            Field(description = "Имя звукового эффекта"),
        ),
    )
