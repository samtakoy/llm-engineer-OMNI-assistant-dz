"""
Подбор голоса синтеза под манеру рассказчика.

pick_voice отдаёт VoiceSettings: имя голоса из переданного списка, темп и
высоту речи. Список голосов приходит снаружи, имена в коде не хранятся.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..integrations.speaking import NO_EFFECT, VoiceSettings, effect_catalog
from .prompts import VOICE_PROMPT

logger = logging.getLogger(__name__)


def pick_voice(
    llm: ChatOpenAI,
    narrator_prompt: str,
    speakers: list[str],
    callbacks: list[BaseCallbackHandler],
) -> tuple[VoiceSettings | None, str]:
    """
    Подбирает голос, темп, высоту речи и звуковой эффект под манеру рассказчика.

    Имя голоса и имя эффекта из ответа модели проверяются по спискам. Голос вне
    списка заменяется первым голосом, эффект вне реестра - отсутствием эффекта.
    Замена печатается.

    Аргументы:
        llm: клиент текстовой модели.
        narrator_prompt: блок про рассказчика.
        speakers: имена голосов, которые знает модель синтеза.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Пару «настройки голоса, причина неудачи». При успехе причина пустая,
        при неудаче настройки None.
    """
    if not speakers:
        return None, "список голосов пуст"

    catalog = effect_catalog()
    effects = "\n".join(f"- {name}: {description}" for name, description in catalog.items())

    structured_llm = llm.with_structured_output(VoiceSettings, method = "json_schema")
    request = (
        f"Рассказчик:\n{narrator_prompt}\n"
        f"Доступные голоса: {', '.join(speakers)}\n"
        f"Доступные эффекты:\n{effects}"
    )

    try:
        settings = structured_llm.invoke(
            [
                SystemMessage(content = VOICE_PROMPT),
                HumanMessage(content = request),
            ],
            config = {"callbacks": callbacks},
        )
    except Exception as error:
        logger.warning(f"[персона] вызов модели не удался: {type(error).__name__}: {error}")
        return None, f"модель не ответила по схеме: {type(error).__name__}"

    corrections: dict[str, str] = {}

    if settings.speaker not in speakers:
        logger.info(f"[персона] голоса {settings.speaker} нет в списке, берётся {speakers[0]}")
        corrections["speaker"] = speakers[0]

    if settings.effect not in catalog:
        logger.info(f"[персона] эффекта {settings.effect} нет в реестре, берётся {NO_EFFECT}")
        corrections["effect"] = NO_EFFECT

    if corrections:
        settings = settings.model_copy(update = corrections)

    return settings, ""
