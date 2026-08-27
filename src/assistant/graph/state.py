"""
Состояние графа и схема структурированного ответа.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Структурированный ответ ресёрчера."""

    answer: str = Field(description = "Ответ на вопрос пользователя, 2-5 предложений")
    confidence: Literal["высокая", "средняя", "низкая"] = Field(
        description = "Насколько найденные материалы подтверждают ответ"
    )
    sources: list[str] = Field(description = "Адреса страниц, на которых основан ответ")


class ResearchState(TypedDict):
    """
    Состояние прогона.

    Поля:
        question: исходный вопрос пользователя.
        messages: история диалога с моделью, включая вызовы инструментов.
        search_rounds: сколько раундов с вызовом инструментов уже сделано.
        answer: структурированный ответ, появляется на последнем узле.
    """

    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    search_rounds: int
    answer: Answer | None
