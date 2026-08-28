"""
Сборка промпта рассказчика из полей персонажа.
"""

from .prompts import NARRATOR_TEMPLATE
from .schemas import Persona


def render_narrator_prompt(persona: Persona) -> str:
    """
    Собирает блок про рассказчика для системного сообщения узла изложения.

    Аргументы:
        persona: рассказчик, выведенный из облика.

    Возвращает:
        Текст блока с подставленными полями персонажа.
    """
    return NARRATOR_TEMPLATE.format(
        name = persona.name,
        gender = persona.gender,
        character = persona.character,
        address_to_listener = persona.address_to_listener,
        speech_manner = persona.speech_manner,
        favourite_words = ", ".join(persona.favourite_words),
        attitude_to_subject = persona.attitude_to_subject,
    )
