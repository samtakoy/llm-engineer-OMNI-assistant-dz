"""
Граф ресёрчера.

Цикл react на сборе фактов, затем два отдельных узла на выходе:

    START -> agent -(есть вызовы инструментов)-> tools -> agent
               \\-(вызовов нет)-> collect -> compose -> END

Узел agent ищет, collect выжимает из найденного проверяемые факты, compose
излагает их в запрошенном пользователем виде.

Модуль описывает только топологию. Клиенты моделей и инструменты приходят
аргументами: граф не знает, какой провайдер за ними стоит и откуда берутся
настройки. Прогон графа и его укомплектование ведёт пакет graph_runs.
"""

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assistant.graph.contracts import NodeLlms
from assistant.graph.nodes import (
    make_agent_node,
    make_collect_node,
    make_compose_node,
    make_tools_node,
)
from assistant.graph.state import ResearchState


def _route_after_agent(state: ResearchState) -> str:
    """
    Выбирает следующий узел по последнему сообщению модели.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Имя следующего узла: tools или collect.
    """
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "collect"


def build_graph(
    checkpointer: BaseCheckpointSaver | None,
    llms: NodeLlms,
    tools: list[BaseTool],
) -> CompiledStateGraph:
    """
    Собирает и компилирует граф ресёрчера.

    Аргументы:
        checkpointer: хранилище снимков состояния; None - без снимков.
        llms: клиенты моделей по узлам.
        tools: инструменты, доступные узлу agent.

    Возвращает:
        Скомпилированный граф, готовый к invoke.
    """
    builder = StateGraph(ResearchState)

    builder.add_node("agent", make_agent_node(llm = llms.agent, tools = tools))
    builder.add_node("tools", make_tools_node(tools = tools))
    builder.add_node("collect", make_collect_node(llm = llms.collect))
    builder.add_node("compose", make_compose_node(llm = llms.compose))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent, ["tools", "collect"])
    builder.add_edge("tools", "agent")
    builder.add_edge("collect", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer = checkpointer)
