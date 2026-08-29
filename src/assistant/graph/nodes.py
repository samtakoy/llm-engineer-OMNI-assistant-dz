"""
Узлы графа.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from assistant.graph.budget import (
    MAX_FAILED_CALLS,
    MAX_SUCCESSFUL_CALLS_PER_TOOL,
    available_tools,
    budget_note,
    count_tool_calls,
)
from assistant.graph.llms import build_agent_llm, build_collect_llm, build_compose_llm
from assistant.graph.logs import (
    log_answer,
    log_blocked_call,
    log_budget,
    log_decision,
    log_narrator_style,
    log_notes,
    log_round,
)
from assistant.graph.prompts import COLLECT_PROMPT, COMPOSE_PROMPT, RESEARCHER_SYSTEM_PROMPT
from assistant.graph.state import Answer, ResearchNotes, ResearchState
from assistant.graph.tools import CALL_BLOCKED, RESEARCH_TOOLS

# Исполнитель инструментов.
_TOOL_EXECUTOR = ToolNode(RESEARCH_TOOLS)


def agent_node(state: ResearchState) -> dict:
    """
    Решает, какой инструмент вызвать, или объявляет, что материала достаточно.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния: ответ модели.
    """
    successful_calls, failed_calls = count_tool_calls(messages = state["messages"])
    tools_left = available_tools(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
    )

    round_number = sum(1 for message in state["messages"] if isinstance(message, AIMessage)) + 1
    log_round(round_number = round_number)
    log_budget(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
        tools_left = tools_left,
    )

    llm = build_agent_llm()
    if tools_left:
        llm = llm.bind_tools(tools_left)

    note = budget_note(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
        tools_left = tools_left,
    )

    # Заметка о бюджете дописывается в первый system: шаблон модели принимает
    # system только первым сообщением.
    message = llm.invoke(
        [
            SystemMessage(content = f"{RESEARCHER_SYSTEM_PROMPT}\n\n{note}"),
            *state["messages"],
        ]
    )

    log_decision(message = message)

    return {"messages": [message]}


def tools_node(state: ResearchState) -> dict:
    """
    Выполняет вызовы инструментов, отсекая те, на которые не осталось бюджета.

    Бюджет тратится по одному вызову, включая пачку параллельных. Ответ
    возвращается на каждый вызов, в том числе на отклонённый.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния: по одному сообщению на каждый вызов.
    """
    successful_calls, failed_calls = count_tool_calls(messages = state["messages"])
    remaining_calls = {
        name: MAX_SUCCESSFUL_CALLS_PER_TOOL - count
        for name, count in successful_calls.items()
    }
    budget_left = failed_calls < MAX_FAILED_CALLS

    last_message = state["messages"][-1]
    allowed_calls, blocked_messages = [], []

    for call in last_message.tool_calls:
        # Неизвестное имя идёт к исполнителю: его ошибка зачтётся в провалы.
        is_unknown_tool = call["name"] not in remaining_calls

        if is_unknown_tool or (budget_left and remaining_calls[call["name"]] > 0):
            if not is_unknown_tool:
                remaining_calls[call["name"]] -= 1
            allowed_calls.append(call)
            continue

        reason = (
            "лимит неудачных вызовов исчерпан"
            if not budget_left
            else "лимит вызовов этого инструмента исчерпан"
        )
        log_blocked_call(tool_name = call["name"], reason = reason)
        blocked_messages.append(
            ToolMessage(
                content = (
                    f"Вызов {call['name']} не выполнен: {reason}. "
                    "Отвечай по уже собранным материалам."
                ),
                name = call["name"],
                tool_call_id = call["id"],
                artifact = CALL_BLOCKED,
            )
        )

    executed_messages = []
    if allowed_calls:
        executed = _TOOL_EXECUTOR.invoke(
            {"messages": [last_message.model_copy(update = {"tool_calls": allowed_calls})]}
        )
        executed_messages = executed["messages"]

    # Порядок ответов возвращается к порядку вызовов.
    by_call_id = {
        message.tool_call_id: message
        for message in [*executed_messages, *blocked_messages]
    }
    return {"messages": [by_call_id[call["id"]] for call in last_message.tool_calls]}


def collect_node(state: ResearchState) -> dict:
    """
    Выжимает из найденных материалов проверяемые факты.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния с полем notes.
    """
    llm = build_collect_llm().with_structured_output(ResearchNotes, method = "json_schema")

    notes = llm.invoke(
        [
            SystemMessage(content = RESEARCHER_SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage(content = COLLECT_PROMPT),
        ]
    )

    log_notes(notes = notes)
    return {"notes": notes}


def compose_node(state: ResearchState) -> dict:
    """
    Излагает собранные факты в виде, который запросил пользователь.

    В контекст уходит только запрос и выжимка фактов, без истории поиска.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния с полем answer.
    """
    notes = state["notes"]
    facts = "\n".join(f"- {fact}" for fact in notes.facts)

    narrator_prompt = state["narrator_prompt"]
    system_prompt = COMPOSE_PROMPT.format(
        style = narrator_prompt if narrator_prompt else "Нейтральный"
    )

    llm = build_compose_llm().with_structured_output(Answer, method = "json_schema")

    answer = llm.invoke(
        [
            SystemMessage(content = system_prompt),
            HumanMessage(
                content = (
                    f"Запрос пользователя:\n{state['question']}\n\n"
                    f"Сводка найденного:\n{notes.summary}\n\n"
                    f"Собранные факты:\n{facts}\n\n"
                    "Весь текст ответа - заголовки, вступление, разделы и "
                    "завершение - пиши на языке запроса пользователя."
                )
            ),
        ]
    )

    log_narrator_style(narrator_prompt)
    log_answer(answer = answer)
    return {"answer": answer}
