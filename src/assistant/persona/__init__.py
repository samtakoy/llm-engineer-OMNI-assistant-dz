"""
Персонаж с фотографии: облик, характер и голос рассказчика.
"""

from .narrator import render_narrator_prompt
from .pipeline import build_narrator_style, build_persona, describe_look
from .prompts import (
    LOOK_PROMPT,
    NARRATOR_STYLE_PROMPT,
    NARRATOR_TEMPLATE,
    PERSONA_PROMPT,
)
from .schemas import Persona, PersonaMode

__all__ = [
    "describe_look",
    "build_persona",
    "build_narrator_style",
    "render_narrator_prompt",
    "Persona",
    "PersonaMode",
    "LOOK_PROMPT",
    "PERSONA_PROMPT",
    "NARRATOR_TEMPLATE",
    "NARRATOR_STYLE_PROMPT",
]
