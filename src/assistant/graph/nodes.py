"""
Узлы графа.

Каждый узел собирается фабрикой: клиент модели и инструменты приходят
аргументами и остаются в замыкании, а сам узел принимает только состояние.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from assistant.graph.budget import (
    MAX_FAILED_CALLS,
    MAX_SUCCESSFUL_CALLS_PER_TOOL,
    available_tools,
    budget_note,
    count_tool_calls,
)
from assistant.graph.contracts import ResearchNode
from assistant.graph.logs import (
    log_answer,
    log_blocked_call,
    log_budget,
    log_decision,
    log_finish_reason,
    log_narrator_style,
    log_notes,
    log_round,
)
from assistant.graph.prompts import (
    COLLECT_PROMPT,
    COLLECT_SYSTEM_PROMPT,
    COMPOSE_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
)
from assistant.graph.state import Answer, ResearchNotes, ResearchState
from assistant.graph.tools import CALL_BLOCKED
from assistant.graph.utils import (
    CAPS_SHARE_THRESHOLD,
    build_case_reference,
    normalize_caps,
)


def make_agent_node(llm: BaseChatModel, tools: list[BaseTool]) -> ResearchNode:
    """
    Собирает узел, решающий, какой инструмент вызвать.

        llm: клиент модели узла.
        tools: инструменты, которыми укомплектован граф.

    Возвращает: Узел графа.
    """

    def agent_node(state: ResearchState) -> dict:
        """
        Решает, какой инструмент вызвать, или объявляет, что материала достаточно.

            state: текущее состояние графа.

        Возвращает: Обновление состояния: ответ модели.
        """
        successful_calls, failed_calls = count_tool_calls(
            messages = state["messages"],
            tools = tools,
        )
        tools_left = available_tools(
            successful_calls = successful_calls,
            failed_calls = failed_calls,
            tools = tools,
        )

        round_number = sum(1 for message in state["messages"] if isinstance(message, AIMessage)) + 1
        log_round(round_number = round_number)
        log_budget(
            successful_calls = successful_calls,
            failed_calls = failed_calls,
            tools_left = tools_left,
        )

        bound_llm = llm.bind_tools(tools_left) if tools_left else llm

        note = budget_note(
            successful_calls = successful_calls,
            failed_calls = failed_calls,
            tools_left = tools_left,
            tools = tools,
        )

        # Заметка о бюджете дописывается в первый system: шаблон модели принимает
        # system только первым сообщением.
        message = bound_llm.invoke(
            [
                SystemMessage(content = f"{RESEARCHER_SYSTEM_PROMPT}\n\n{note}"),
                *state["messages"],
            ]
        )

        log_finish_reason(message = message)
        log_decision(message = message)

        return {"messages": [message]}

    return agent_node


def make_tools_node(tools: list[BaseTool]) -> ResearchNode:
    """
    Собирает узел, выполняющий вызовы инструментов.

        tools: инструменты, которыми укомплектован граф.

    Возвращает: Узел графа.
    """
    executor = ToolNode(tools)

    def tools_node(state: ResearchState) -> dict:
        """
        Выполняет вызовы инструментов, отсекая те, на которые не осталось бюджета.

        Бюджет тратится по одному вызову, включая пачку параллельных. Ответ
        возвращается на каждый вызов, в том числе на отклонённый.

            state: текущее состояние графа.

        Возвращает: Обновление состояния - по одному сообщению на каждый вызов.
        """
        successful_calls, failed_calls = count_tool_calls(
            messages = state["messages"],
            tools = tools,
        )
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
            executed = executor.invoke(
                {"messages": [last_message.model_copy(update = {"tool_calls": allowed_calls})]}
            )
            executed_messages = executed["messages"]

        # Порядок ответов возвращается к порядку вызовов.
        by_call_id = {
            message.tool_call_id: message
            for message in [*executed_messages, *blocked_messages]
        }
        return {"messages": [by_call_id[call["id"]] for call in last_message.tool_calls]}

    return tools_node


def make_collect_node(llm: BaseChatModel) -> ResearchNode:
    """
    Собирает узел, выжимающий факты из найденных материалов.

        llm: клиент модели узла.

    Возвращает: Узел графа.
    """
    structured_llm = llm.with_structured_output(ResearchNotes, method = "json_schema")

    def collect_node(state: ResearchState) -> dict:
        """
        Выжимает из найденных материалов проверяемые факты.

            state: текущее состояние графа.

        Возвращает: Обновление состояния с полем notes.
        """
        notes = structured_llm.invoke(
            [
                SystemMessage(content = COLLECT_SYSTEM_PROMPT),
                *state["messages"],
                HumanMessage(content = COLLECT_PROMPT),
            ]
        )

        log_notes(notes = notes)
        return {"notes": notes}

    return collect_node


def _as_list_block(items: list[str], empty_text: str) -> str:
    """
    Собирает список строк в блок опоры для узла изложения.

        items: строки опоры.
        empty_text: текст, который встанет вместо пустого списка.

    Возвращает: Строки списком через перенос или empty_text, если список пуст.
    """
    if not items:
        return empty_text

    return "\n".join(f"- {item}" for item in items)


def _without_caps(answer: Answer) -> Answer:
    """Убирает капслок из заголовков и текста ответа."""

    # Сборка словаря имен собственных
    case_reference = build_case_reference(
        text = "\n".join(
            [
                answer.title,
                answer.intro,
                *[f"{section.title}\n{section.content}" for section in answer.sections],
                answer.closing,
            ]
        )
    )

    def normalize(text: str) -> str:
        """Нормализует одну строку ответа общими образцами и порогом."""
        return normalize_caps(
            text = text,
            case_reference = case_reference,
            caps_threshold = CAPS_SHARE_THRESHOLD,
        )

    return answer.model_copy(
        update = {
            "title": normalize(text = answer.title),
            "intro": normalize(text = answer.intro),
            "sections": [
                section.model_copy(
                    update = {
                        "title": normalize(text = section.title),
                        "content": normalize(text = section.content),
                    }
                )
                for section in answer.sections
            ],
            "closing": normalize(text = answer.closing),
        }
    )


def make_compose_node(llm: BaseChatModel) -> ResearchNode:
    """
    Собирает узел, излагающий собранные факты.

        llm: клиент модели узла.

    Возвращает: Узел графа.
    """
    structured_llm = llm.with_structured_output(Answer, method = "json_schema")

    def compose_node(state: ResearchState) -> dict:
        """
        Излагает собранные факты в виде, который запросил пользователь.

        В контекст уходит запрос и опора, собранная узлом collect, без истории
        поиска. Опора идёт размеченными блоками, задача - последней строкой.

            state: текущее состояние графа.

        Возвращает: Обновление состояния с полем answer.
        """
        notes = state["notes"]

        narrator_prompt = state["narrator_prompt"]
        system_prompt = COMPOSE_PROMPT.format(
            style = narrator_prompt if narrator_prompt else "Нейтральный"
        )

        answer = structured_llm.invoke(
            [
                SystemMessage(content = system_prompt),
                HumanMessage(
                    content = (
                        f"<запрос>\n{state['question']}\n</запрос>\n\n"
                        f"<порядок изложения>\n{notes.summary}\n</порядок изложения>\n\n"
                        "<опора>\n"
                        f"{_as_list_block(items = notes.facts, empty_text = 'фактов нет')}\n"
                        "</опора>\n\n"
                        "<подробности>\n"
                        f"{_as_list_block(items = notes.details, empty_text = 'подробностей нет')}\n"
                        "</подробности>\n\n"
                        "<чего не нашлось>\n"
                        f"{_as_list_block(items = notes.gaps, empty_text = 'пропусков не отмечено')}\n"
                        "</чего не нашлось>\n\n"
                        "<заметки>\n"
                        f"{notes.handoff if notes.handoff else 'заметок нет'}\n"
                        "</заметки>\n\n"
                        f"Достоверность опоры: {notes.confidence}.\n\n"
                        "Задача: изложи опору по запросу пользователя "
                        "с учетом правил, запретов и указанной роли."
                    )
                ),
            ]
        )

        answer = _without_caps(answer = answer)

        log_narrator_style(narrator_prompt)
        log_answer(answer = answer)
        return {"answer": answer}

    return compose_node
