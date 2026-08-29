"""
Граф ресёрчера.

Цикл react на сборе фактов, затем два отдельных узла на выходе:

    START -> agent -(есть вызовы инструментов)-> tools -> agent
               \\-(вызовов нет)-> collect -> compose -> END

Узел agent ищет, collect выжимает из найденного проверяемые факты, compose
излагает их в запрошенном пользователем виде.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from assistant.graph.budget import MAX_TOOL_CALLS_PER_RUN
from assistant.graph.nodes import agent_node, collect_node, compose_node, tools_node
from assistant.graph.state import Answer, ResearchNotes, ResearchState


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


def build_graph():
    """
    Собирает и компилирует граф ресёрчера.

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

    return builder.compile()


def run_research(question: str, narrator_prompt: str | None) -> tuple[Answer, ResearchNotes]:
    """
    Прогоняет вопрос через граф.

    Аргументы:
        question: вопрос пользователя.
        narrator_prompt: блок про рассказчика для узла изложения; None -
            изложение без персонажа.

    Возвращает:
        Кортеж из итогового текста и фактической опоры, на которой он построен.
    """
    initial_state: ResearchState = {
        "question": question,
        "messages": [HumanMessage(content = question)],
        "narrator_prompt": narrator_prompt,
        "notes": None,
        "answer": None,
    }

    # Запас по рекурсии: раунд стоит два шага (agent + tools), плюс финальный
    # agent и два узла вывода.
    final_state = build_graph().invoke(
        initial_state,
        config = {"recursion_limit": MAX_TOOL_CALLS_PER_RUN * 2 + 5},
    )

    return final_state["answer"], final_state["notes"]
