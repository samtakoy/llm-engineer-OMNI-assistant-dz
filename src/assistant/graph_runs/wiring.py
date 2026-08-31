"""
Укомплектование графа для этого проекта.

Единственное место, где ядро графа встречается с клиентами моделей: ниже граф
не знает, какой за ними провайдер, выше о них не думают. Инструменты у графа
свои, они приходят из его же пакета.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from assistant.graph import RESEARCH_TOOLS, build_graph
from assistant.graph_runs.llms import build_node_llms


def build_research_graph(checkpointer: BaseCheckpointSaver | None) -> CompiledStateGraph:
    """
    Собирает граф ресёрчера с клиентами моделей этого проекта.

    Аргументы:
        checkpointer: хранилище снимков состояния; None - без снимков.

    Возвращает:
        Скомпилированный граф, готовый к invoke.
    """
    return build_graph(
        checkpointer = checkpointer,
        llms = build_node_llms(),
        tools = RESEARCH_TOOLS,
    )
