"""
Персонаж с фотографии: облик, характер и голос рассказчика.

Пакет считается до графа и графом не пользуется: граф отвечает на вопрос «что
известно и откуда», персона - на вопрос «кто это рассказывает».

Ни одна публичная функция не выбрасывает исключений наружу - при неудаче
возвращается краткая причина.
"""

from .narrator import render_narrator_prompt
from .pipeline import build_persona, describe_look
from .prompts import LOOK_PROMPT, NARRATOR_TEMPLATE, PERSONA_PROMPT
from .schemas import Persona

__all__ = [
    "describe_look",
    "build_persona",
    "render_narrator_prompt",
    "Persona",
    "LOOK_PROMPT",
    "PERSONA_PROMPT",
    "NARRATOR_TEMPLATE",
]
