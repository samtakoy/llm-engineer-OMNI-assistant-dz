"""
Подбор голоса синтеза под манеру рассказчика.

pick_voice отдаёт VoiceChoice: имя голоса из переданного списка, темп и
высоту речи, а рядом пол рассказчика. Список голосов и запасные голоса по полу
приходят снаружи, имена в коде не хранятся.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..integrations.speaking import NO_EFFECT, effect_catalog
from .prompts import VOICE_PROMPT
from .schemas import VoiceChoice, voice_choice_schema

logger = logging.getLogger(__name__)


def _fallback_speaker(
    narrator_gender: str,
    speakers: list[str],
    speakers_by_gender: dict[str, str],
) -> str:
    """
    Подбирает запасной голос, когда модель назвала голос вне списка.

    Аргументы:
        narrator_gender: пол рассказчика из ответа модели.
        speakers: имена голосов, которые знает модель синтеза.
        speakers_by_gender: голоса, чей пол проекту известен, по полу.

    Возвращает:
        Голос своего пола, если проект такой знает и модель синтеза его умеет,
        иначе первый голос списка.
    """
    speaker = speakers_by_gender.get(narrator_gender, "")

    if speaker in speakers:
        return speaker

    return speakers[0]


def pick_voice(
    llm: ChatOpenAI,
    narrator_prompt: str,
    speakers: list[str],
    speakers_by_gender: dict[str, str],
    callbacks: list[BaseCallbackHandler],
) -> tuple[VoiceChoice | None, str]:
    """
    Подбирает голос, темп, высоту речи и звуковой эффект под манеру рассказчика.

    Схема ответа собирается под переданные списки: имя голоса и имя эффекта
    уезжают в запрос перечислениями. Ответ вне списков проверяется ещё раз, уже
    после разбора. Голос вне списка заменяется голосом своего пола, эффект вне реестра - отсутствием
    эффекта. Замена печатается.

    Аргументы:
        llm: клиент текстовой модели.
        narrator_prompt: блок про рассказчика.
        speakers: имена голосов, которые знает модель синтеза.
        speakers_by_gender: голоса, чей пол проекту известен, по полу.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Пару «настройки голоса, причина неудачи». При успехе причина пустая,
        при неудаче настройки None.
    """
    if not speakers:
        return None, "список голосов пуст"

    catalog = effect_catalog()
    effects = "\n".join(f"- {name}: {description}" for name, description in catalog.items())

    schema = voice_choice_schema(
        speakers = tuple(speakers),
        effects = tuple(catalog),
    )
    structured_llm = llm.with_structured_output(schema, method = "json_schema")
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

    if settings.voice_name not in speakers:
        replacement = _fallback_speaker(
            narrator_gender = settings.narrator_gender,
            speakers = speakers,
            speakers_by_gender = speakers_by_gender,
        )
        logger.info(f"[персона] голоса {settings.voice_name} нет в списке, берётся {replacement}")
        corrections["voice_name"] = replacement

    if settings.effect not in catalog:
        logger.info(f"[персона] эффекта {settings.effect} нет в реестре, берётся {NO_EFFECT}")
        corrections["effect"] = NO_EFFECT

    if corrections:
        settings = settings.model_copy(update = corrections)

    return settings, ""
