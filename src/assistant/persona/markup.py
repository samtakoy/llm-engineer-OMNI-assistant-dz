"""
Разметка текста ssml под манеру речи персонажа.

mark_up_speech отдаёт модели текст и манеру рассказчика, получает тот же текст
с паузами, ударениями и звуками персонажа. В начало ответа добавляется пауза:
без неё голос вступает обрывисто. Чистка разметки идёт при синтезе, здесь ответ
модели не проверяется.

Наружу исключения не уходят - причина возвращается второй половиной пары.
"""

import logging
import re

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .prompts import MARKUP_PROMPT

logger = logging.getLogger(__name__)


# Пауза перед первым словом.
LEAD_PAUSE = '<break time="300ms"/>'

# Пауза, уже стоящая в начале ответа модели.
_LEADING_BREAK_PATTERN = re.compile(r"^\s*<\s*break[^>]*>", re.IGNORECASE)


def mark_up_speech(
    llm: ChatOpenAI,
    narrator_prompt: str,
    text: str,
    callbacks: list[BaseCallbackHandler],
) -> tuple[str, str]:
    """
    Размечает текст паузами, ударениями и звуками под манеру рассказчика.

    Аргументы:
        llm: клиент текстовой модели.
        narrator_prompt: блок про рассказчика.
        text: текст без разметки.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Пару «размеченный текст, причина неудачи». Размеченный текст начинается
        паузой. При успехе причина пустая, при неудаче текст пустой.
    """
    if not text.strip():
        return "", "размечать нечего"

    request = (
        f"Рассказчик:\n{narrator_prompt}\n"
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

    return _with_lead_pause(marked = marked), ""


def _with_lead_pause(marked: str) -> str:
    """
    Ставит паузу перед первым словом размеченного текста.

    Аргументы:
        marked: размеченный текст от модели.

    Возвращает:
        Текст, начинающийся паузой. Своя пауза модели остаётся единственной.
    """
    if _LEADING_BREAK_PATTERN.match(marked):
        return marked

    return f"{LEAD_PAUSE}{marked}"
