"""
Схемы структурированного вывода пакета персонажа.
"""

from typing import Literal

from pydantic import BaseModel, Field


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
    attitude_to_subject: str = Field(
        description = "Как персонаж относится к предмету рассказа и что в нём выделяет"
    )
