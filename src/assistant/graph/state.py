"""
Состояние графа и схемы структурированного вывода.

Схем две, потому что этапа два. Ресёрч отвечает на вопрос «что известно и
откуда», изложение - на вопрос «в каком виде это подать». Их разделение не даёт
требованию стиля протекать в сбор фактов и наоборот.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ResearchNotes(BaseModel):
    """Фактическая опора, собранная по источникам."""

    summary: str = Field(description = "Краткая сводка найденного")
    facts: list[str] = Field(
        description = "Отдельные факты, каждый одним предложением, только из найденных материалов"
    )
    confidence: Literal["высокая", "средняя", "низкая"] = Field(
        description = "Насколько найденные материалы подтверждают собранное"
    )
    sources: list[str] = Field(description = "Адреса страниц, на которых основаны факты")


class Section(BaseModel):
    """Один раздел итогового текста."""

    title: str = Field(description = "Заголовок раздела на языке запроса пользователя")
    content: str = Field(
        description = "Текст раздела на языке запроса, в запрошенном пользователем стиле"
    )


class Answer(BaseModel):
    """
    Итоговый текст в том виде, который запросил пользователь.

    Схема намеренно нейтральна: разделами одинаково ложатся точки маршрута,
    шаги инструкции и пункты разбора. Что именно окажется разделом, решает
    запрос, а не код.
    """

    title: str = Field(description = "Заголовок всего текста на языке запроса пользователя")
    intro: str = Field(description = "Вступление на языке запроса пользователя")
    sections: list[Section] = Field(description = "Разделы по порядку изложения")
    closing: str = Field(description = "Завершение на языке запроса пользователя")


class ResearchState(TypedDict):
    """
    Состояние прогона.

    Отдельных счётчиков вызовов тут нет намеренно: бюджет инструментов
    считается из messages, где каждый вызов уже записан вместе с его исходом.
    Второй источник истины разошёлся бы с историей при любой правке графа.

    Поля:
        question: исходный вопрос пользователя.
        messages: история диалога с моделью, включая вызовы инструментов.
        notes: фактическая опора, появляется на узле collect.
        answer: итоговый текст, появляется на узле compose.
    """

    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    notes: ResearchNotes | None
    answer: Answer | None
