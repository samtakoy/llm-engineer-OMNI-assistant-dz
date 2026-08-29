"""
Сборка всех этапов: фотография персонажа, вопрос, экскурсия его голосом.

Точки входа зовут только эту функцию: командная строка и интерфейс идут одним
путём.
"""

from dataclasses import dataclass
from pathlib import Path

from assistant.graph import Answer, ResearchNotes, run_research
from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.persona import (
    Persona,
    PersonaMode,
    build_narrator_style,
    build_persona,
    describe_look,
    render_narrator_prompt,
)
from assistant.variables import ENABLE_ALL_REASONING, VISION_MODEL, VISION_PROVIDER


@dataclass(frozen = True)
class OmniOutcome:
    """
    Исход прогона.

    Атрибуты:
        answer: итоговый текст экскурсии; None при неудаче.
        notes: фактическая опора, на которой построен текст; None при неудаче.
        persona: рассказчик полями; заполняется только в режиме STRUCTURED.
        narrator_prompt: блок про рассказчика, ушедший в узел изложения; пустая
            строка, если рассказчик не задан.
        look: описание облика с фотографии; пустая строка, если фотографии не было.
        error: причина неудачи; пустая строка при успехе.
    """

    answer: Answer | None
    notes: ResearchNotes | None
    persona: Persona | None
    narrator_prompt: str
    look: str
    error: str


def build_narrator_from_image(
    image_path: Path,
    persona_mode: PersonaMode,
) -> tuple[str, Persona | None, str, str]:
    """
    Строит блок про рассказчика по фотографии персонажа.

    Аргументы:
        image_path: файл с фотографией персонажа.
        persona_mode: способ сборки: одной фразой либо полями схемы.

    Возвращает:
        Четвёрку «блок про рассказчика, персонаж полями, описание облика,
        причина неудачи». Персонаж заполняется только в режиме STRUCTURED.
        При неудаче блок пустой.
    """
    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )

    look, error = describe_look(llm = vision_llm, image_path = image_path)
    if error:
        return "", None, "", error

    writing_llm = build_llm(
        role = NodeRole.WRITING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )


    if persona_mode is PersonaMode.FREE:
        # свободное изложение описания
        style, error = build_narrator_style(llm = writing_llm, look = look)
        if error:
            return "", None, look, error

        print(f"[рассказчик]\n{style}")
        return style, None, look, ""

    # описание рассказчика в виде структурированного разложения по осям
    persona, error = build_persona(llm = writing_llm, look = look)
    if error:
        return "", None, look, error

    print(f"[рассказчик] {persona.name}: {persona.speech_manner}")
    return render_narrator_prompt(persona = persona), persona, look, ""


def run_omni_assistant(
    image_path: Path | None,
    narrator_style: str | None,
    persona_mode: PersonaMode,
    question: str,
) -> OmniOutcome:
    """
    Проводит экскурсию по вопросу от лица заданного рассказчика.

    Рассказчик берётся из фотографии либо из готовой фразы. Без того и другого
    текст пишется обычным рассказчиком.

    Аргументы:
        image_path: файл с фотографией персонажа; None - без фотографии.
        narrator_style: готовая фраза про голос рассказчика; None - без неё.
        persona_mode: способ сборки рассказчика по фотографии.
        question: вопрос пользователя.

    Возвращает:
        Исход прогона: текст экскурсии с опорой либо причина неудачи.
    """
    persona: Persona | None = None
    narrator_prompt = ""
    look = ""

    if narrator_style is not None:
        narrator_prompt = narrator_style
        print(f"[рассказчик]\n{narrator_prompt}")
    elif image_path is not None:
        narrator_prompt, persona, look, error = build_narrator_from_image(
            image_path = image_path,
            persona_mode = persona_mode,
        )
        if error:
            return OmniOutcome(
                answer = None,
                notes = None,
                persona = None,
                narrator_prompt = "",
                look = look,
                error = error,
            )

    answer, notes = run_research(
        question = question,
        narrator_prompt = narrator_prompt or None,
    )

    return OmniOutcome(
        answer = answer,
        notes = notes,
        persona = persona,
        narrator_prompt = narrator_prompt,
        look = look,
        error = "",
    )
