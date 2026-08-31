"""
Граф ресёрчера.

Цикл react на сборе фактов, затем два отдельных узла на выходе:

    START -> agent -(есть вызовы инструментов)-> tools -> agent
               \\-(вызовов нет)-> collect -> compose -> END

Узел agent ищет, collect выжимает из найденного проверяемые факты, compose
излагает их в запрошенном пользователем виде.

Модуль описывает только топологию. Прогон графа ведёт runs, чтение записанных
снимков - history.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assistant.graph.nodes import agent_node, collect_node, compose_node, tools_node
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


def build_graph(checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
    """
    Собирает и компилирует граф ресёрчера.

    Аргументы:
        checkpointer: хранилище снимков состояния; None - без снимков.

    Возвращает:
        Скомпилированный граф, готовый к invoke.
    """
    builder = StateGraph(ResearchState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("collect", collect_node)
    builder.add_node("compose", compose_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent, ["tools", "collect"])
    builder.add_edge("tools", "agent")
    builder.add_edge("collect", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer = checkpointer)
