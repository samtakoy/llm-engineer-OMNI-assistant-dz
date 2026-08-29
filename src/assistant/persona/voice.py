"""
Подбор голоса синтеза под характер персонажа.

pick_voice отдаёт VoiceSettings: имя голоса из переданного списка, темп и
высоту речи. Список голосов приходит снаружи, имена в коде не хранятся.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..integrations.speaking import VoiceSettings
from .narrator import render_narrator_prompt
from .prompts import VOICE_PROMPT
from .schemas import Persona


def pick_voice(
    llm: ChatOpenAI,
    persona: Persona,
    speakers: list[str],
) -> tuple[VoiceSettings | None, str]:
    """
    Подбирает голос, темп и высоту речи под характер персонажа.

    Имя голоса из ответа модели проверяется по списку. Имя вне списка
    заменяется первым голосом списка, замена печатается.

    Аргументы:
        llm: клиент текстовой модели.
        persona: рассказчик, выведенный из облика.
        speakers: имена голосов, которые знает модель синтеза.

    Возвращает:
        Пару «настройки голоса, причина неудачи». При успехе причина пустая,
        при неудаче настройки None.
    """
    if not speakers:
        return None, "список голосов пуст"

    structured_llm = llm.with_structured_output(VoiceSettings, method = "json_schema")
    request = (
        f"Рассказчик:\n{render_narrator_prompt(persona = persona)}\n"
        f"Доступные голоса: {', '.join(speakers)}"
    )

    try:
        settings = structured_llm.invoke(
            [
                SystemMessage(content = VOICE_PROMPT),
                HumanMessage(content = request),
            ]
        )
    except Exception as error:
        print(f"[персона] вызов модели не удался: {type(error).__name__}: {error}")
        return None, f"модель не ответила по схеме: {type(error).__name__}"

    if settings.speaker not in speakers:
        print(f"[персона] голоса {settings.speaker} нет в списке, берётся {speakers[0]}")
        settings = settings.model_copy(update = {"speaker": speakers[0]})

    return settings, ""
