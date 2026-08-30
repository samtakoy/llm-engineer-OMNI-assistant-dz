"""
Персонаж с фотографии: облик, характер и голос рассказчика.
"""

from .markup import mark_up_speech
from .narrator import render_narrator_prompt
from .pipeline import build_narrator_style, build_persona, describe_look
from .prompts import (
    LOOK_PROMPT,
    MARKUP_PROMPT,
    NARRATOR_STYLE_PROMPT,
    NARRATOR_TEMPLATE,
    PERSONA_PROMPT,
    VOICE_PROMPT,
)
from .schemas import NarratorVoice, Persona, PersonaMode
from .voice import pick_voice

__all__ = [
    "describe_look",
    "build_persona",
    "build_narrator_style",
    "render_narrator_prompt",
    "pick_voice",
    "mark_up_speech",
    "Persona",
    "PersonaMode",
    "NarratorVoice",
    "LOOK_PROMPT",
    "PERSONA_PROMPT",
    "NARRATOR_TEMPLATE",
    "NARRATOR_STYLE_PROMPT",
    "VOICE_PROMPT",
    "MARKUP_PROMPT",
]
