"""
Сборка клиентов моделей по узлам графа.

Каждый узел получает клиент по своей роли NodeRole. Параметры сэмплирования
для роли хранит модуль profiles.
"""

from langchain_openai import ChatOpenAI

from assistant.integrations.llm.client import build_llm, describe_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.variables import ENABLE_ALL_REASONING


def build_agent_llm() -> ChatOpenAI:
    """
    Собирает клиент узла agent.

    Возвращает:
        Клиент модели.
    """
    return build_llm(
        role = NodeRole.TOOL_CALLING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )


def build_collect_llm() -> ChatOpenAI:
    """
    Собирает клиент узла collect без схемы.

    Схема навешивается в самом узле: with_structured_output возвращает уже
    не ChatOpenAI.

    Возвращает:
        Клиент модели.
    """
    return build_llm(
        role = NodeRole.EXTRACTION,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )


def build_compose_llm() -> ChatOpenAI:
    """
    Собирает клиент узла compose без схемы.

    Возвращает:
        Клиент модели.
    """
    return build_llm(
        role = NodeRole.WRITING,
        is_reasoning_forced = ENABLE_ALL_REASONING,
        model = None,
    )


def describe_nodes() -> list[str]:
    """
    Описывает параметры моделей по узлам.

    Возвращает:
        Список строк для вывода в командной строке.
    """
    return [
        f"agent:   {describe_llm(llm = build_agent_llm())}",
        f"collect: {describe_llm(llm = build_collect_llm())}",
        f"compose: {describe_llm(llm = build_compose_llm())}",
    ]
