"""
Сборка клиентов моделей по узлам графа.

Каждый узел получает клиент по своей роли NodeRole. Параметры сэмплирования
для роли хранит модуль profiles. Сам граф клиентов не создаёт: он принимает
готовый набор NodeLlms.
"""

from langchain_openai import ChatOpenAI

from assistant.graph.contracts import NodeLlms
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


def build_node_llms() -> NodeLlms:
    """
    Собирает клиенты всех узлов графа.

    Возвращает:
        Набор клиентов, который принимает build_graph.
    """
    return NodeLlms(
        agent = build_agent_llm(),
        collect = build_collect_llm(),
        compose = build_compose_llm(),
    )


def describe_nodes(llms: NodeLlms) -> list[str]:
    """
    Описывает параметры моделей по узлам.

    Аргументы:
        llms: клиенты узлов графа.

    Возвращает:
        Список строк для вывода в командной строке.
    """
    return [
        f"agent:   {describe_llm(llm = llms.agent)}",
        f"collect: {describe_llm(llm = llms.collect)}",
        f"compose: {describe_llm(llm = llms.compose)}",
    ]
