"""
Разметка текста ssml под манеру речи персонажа.

mark_up_speech отдаёт модели текст и манеру рассказчика, получает тот же текст
с тегами пауз, темпа и ударений. Чистка разметки идёт при синтезе, здесь ответ
модели не проверяется.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..integrations.speaking import pitch_values, rate_values
from .narrator import render_narrator_prompt
from .prompts import MARKUP_PROMPT
from .schemas import Persona

logger = logging.getLogger(__name__)


def mark_up_speech(
    llm: ChatOpenAI,
    persona: Persona,
    text: str,
    callbacks: list[BaseCallbackHandler],
) -> tuple[str, str]:
    """
    Размечает текст паузами, ударениями и сменой темпа под манеру персонажа.

    Аргументы:
        llm: клиент текстовой модели.
        persona: рассказчик, выведенный из облика.
        text: текст без разметки.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Пару «размеченный текст, причина неудачи». При успехе причина пустая,
        при неудаче текст пустой.
    """
    if not text.strip():
        return "", "размечать нечего"

    request = (
        f"Рассказчик:\n{render_narrator_prompt(persona = persona)}\n"
        f"Значения темпа: {', '.join(rate_values())}\n"
        f"Значения высоты: {', '.join(pitch_values())}\n"
        f"Текст:\n{text}"
    )

    try:
        message = llm.invoke(
            [
                SystemMessage(content = MARKUP_PROMPT),
                HumanMessage(content = request),
            ],
            config = {"callbacks": callbacks},
        )
    except Exception as error:
        logger.warning(f"[персона] вызов модели не удался: {type(error).__name__}: {error}")
        return "", f"модель не ответила: {type(error).__name__}"

    marked = message.text.strip()
    if not marked:
        return "", "модель вернула пустой ответ"

    return marked, ""
