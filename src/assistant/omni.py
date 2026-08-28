"""
Сборка всех этапов: фотография персонажа, вопрос, экскурсия его голосом.

Точки входа зовут только эту функцию: командная строка и интерфейс идут одним
путём.
"""

from dataclasses import dataclass
from pathlib import Path

from assistant.graph.graph import run_research
from assistant.graph.state import Answer, ResearchNotes
from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.persona import Persona, build_persona, describe_look, render_narrator_prompt
from assistant.variables import VISION_MODEL, VISION_PROVIDER


@dataclass(frozen = True)
class OmniOutcome:
    """
    Исход прогона.

    Атрибуты:
        answer: итоговый текст экскурсии; None при неудаче.
        notes: фактическая опора, на которой построен текст; None при неудаче.
        persona: рассказчик с фотографии; None, если фотографии не было.
        look: описание облика с фотографии; пустая строка, если фотографии не было.
        error: причина неудачи; пустая строка при успехе.
    """

    answer: Answer | None
    notes: ResearchNotes | None
    persona: Persona | None
    look: str
    error: str


def build_narrator(image_path: Path) -> tuple[Persona | None, str, str]:
    """
    Строит рассказчика по фотографии персонажа.

    Аргументы:
        image_path: файл с фотографией персонажа.

    Возвращает:
        Тройку «персонаж, описание облика, причина неудачи». При успехе причина
        пустая, при неудаче персонаж None.
    """
    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_debug_reasoning_on = False,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )

    look, error = describe_look(llm = vision_llm, image_path = image_path)
    if error:
        return None, "", error

    writing_llm = build_llm(role = NodeRole.WRITING, is_debug_reasoning_on = False, model = None)

    persona, error = build_persona(llm = writing_llm, look = look)
    if error:
        return None, look, error

    return persona, look, ""


def run_omni_assistant(image_path: Path | None, question: str) -> OmniOutcome:
    """
    Проводит экскурсию по вопросу от лица персонажа с фотографии.

    Без фотографии текст пишется обычным рассказчиком.

    Аргументы:
        image_path: файл с фотографией персонажа; None - без персонажа.
        question: вопрос пользователя.

    Возвращает:
        Исход прогона: текст экскурсии с опорой либо причина неудачи.
    """
    persona: Persona | None = None
    look = ""
    narrator_prompt: str | None = None

    if image_path is not None:
        persona, look, error = build_narrator(image_path = image_path)
        # Ресёрч стоит минуты, поэтому неудача разбора роняет прогон здесь, а не
        # после сбора фактов.
        if error:
            return OmniOutcome(answer = None, notes = None, persona = None, look = look, error = error)

        narrator_prompt = render_narrator_prompt(persona = persona)

    answer, notes = run_research(question = question, narrator_prompt = narrator_prompt)

    return OmniOutcome(answer = answer, notes = notes, persona = persona, look = look, error = "")
