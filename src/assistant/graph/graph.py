"""
Граф ресёрчера.

Цикл react на сборе фактов, затем два отдельных узла на выходе:

    START -> agent -(есть вызовы инструментов)-> tools -> agent
               \\-(вызовов нет)-> collect -> compose -> END

Этапы разделены намеренно. Узел agent ищет, collect выжимает из найденного
проверяемые факты, compose излагает их в запрошенном пользователем виде. Пока
сбор и изложение жили в одном узле, требование стиля тянуло модель прочь от
источников, а требование опираться на источники гасило стиль.

Структура собирается отдельными вызовами, а не тем же, что и вызовы
инструментов: локальные модели плохо переносят совмещение tool calling и
жёсткой схемы в одном запросе.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from assistant.graph.prompts import (
    COLLECT_PROMPT,
    COMPOSE_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
)
from assistant.graph.state import Answer, ResearchNotes, ResearchState
from assistant.graph.tools import RESEARCH_TOOLS
from assistant.integrations.llm.client import build_llm, describe_llm, reasoning_text
from assistant.variables import SHOW_REASONING

# Сколько раундов с вызовом инструментов разрешено до принудительного ответа.
MAX_SEARCH_ROUNDS = 6

# Температура не задаётся: её берём из профиля модели, он тут источник истины.
_TEMPERATURE_FROM_PROFILE = None

# Бюджет размышления узла agent. Значение low обязательно: у qwen бюджет по
# умолчанию не ограничен, и на вызовах инструментов модель уходит в рассуждение
# на десятки тысяч токенов и не останавливается.
#
# Само размышление включается переменной окружения SHOW_REASONING. Без неё
# профиль гасит его совсем: платить временем за то, чего не видно, незачем.
_AGENT_REASONING_EFFORT = "low"

# Узлы вывода работают под грамматикой, а под ней размышление оставляет content
# пустым. Гасит это только "none", low и minimal не помогают.
_OUTPUT_REASONING_EFFORT = "none"

# Потолок ответа узла agent: вызов инструмента короткий, финальная реплика тоже.
# Страховка на случай, если модель всё-таки зациклится.
_AGENT_MAX_TOKENS = 9000


def _build_agent_llm() -> ChatOpenAI:
    """
    Собирает клиент узла agent.

    Возвращает:
        Клиент модели.
    """
    return build_llm(
        temperature = _TEMPERATURE_FROM_PROFILE,
        reasoning_effort = _AGENT_REASONING_EFFORT,
        max_tokens = _AGENT_MAX_TOKENS,
        show_reasoning = SHOW_REASONING,
    )


def _build_output_llm() -> ChatOpenAI:
    """
    Собирает клиент узлов вывода без схемы.

    Схема навешивается в самих узлах: описанию в логе она не нужна, а клиент
    после with_structured_output перестаёт быть ChatOpenAI.

    Возвращает:
        Клиент модели.
    """
    return build_llm(
        temperature = _TEMPERATURE_FROM_PROFILE,
        reasoning_effort = _OUTPUT_REASONING_EFFORT,
        max_tokens = None,
        show_reasoning = False,
    )


def describe_nodes() -> list[str]:
    """
    Описывает параметры моделей по узлам.

    Возвращает:
        Список строк для вывода в командной строке.
    """
    return [
        f"agent:            {describe_llm(llm = _build_agent_llm())}",
        f"collect, compose: {describe_llm(llm = _build_output_llm())}",
    ]


def _log_reasoning(message: AIMessage, round_number: int) -> None:
    """
    Печатает размышление модели и её решение на текущем шаге.

    Аргументы:
        message: ответ модели.
        round_number: номер раунда, для читаемости лога.

    Возвращает:
        Ничего.
    """
    print(f"\n--- раунд {round_number} ---")

    reasoning = reasoning_text(message)
    if reasoning:
        print(f"[размышление]\n{reasoning}")

    if message.tool_calls:
        for call in message.tool_calls:
            print(f"[инструмент] {call['name']}({call['args']})")
    else:
        print("[решение] инструменты больше не нужны, перехожу к сбору фактов")


def _agent_node(state: ResearchState) -> dict:
    """
    Решает, какой инструмент вызвать, или объявляет, что материала достаточно.

    На последнем разрешённом раунде модель вызывается без инструментов: так она
    физически не может запросить ещё один поиск, и граф гарантированно доходит
    до ответа, не оставляя вызовов без ответа.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния: ответ модели и счётчик раундов.
    """
    limit_reached = state["search_rounds"] >= MAX_SEARCH_ROUNDS
    llm = _build_agent_llm()

    if not limit_reached:
        llm = llm.bind_tools(RESEARCH_TOOLS)

    message = llm.invoke(
        [SystemMessage(content = RESEARCHER_SYSTEM_PROMPT), *state["messages"]]
    )

    _log_reasoning(message = message, round_number = state["search_rounds"] + 1)

    made_tool_calls = bool(getattr(message, "tool_calls", None))
    return {
        "messages": [message],
        "search_rounds": state["search_rounds"] + (1 if made_tool_calls else 0),
    }


def _collect_node(state: ResearchState) -> dict:
    """
    Выжимает из найденных материалов проверяемые факты.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния с полем notes.
    """
    llm = _build_output_llm().with_structured_output(ResearchNotes, method = "json_schema")

    notes = llm.invoke(
        [
            SystemMessage(content = RESEARCHER_SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage(content = COLLECT_PROMPT),
        ]
    )

    print(f"\n[факты] собрано {len(notes.facts)}, источников {len(notes.sources)}")
    return {"notes": notes}


def _compose_node(state: ResearchState) -> dict:
    """
    Излагает собранные факты в виде, который запросил пользователь.

    В контекст уходит только запрос и выжимка фактов, без истории поиска: сырые
    страницы тянут модель в пересказ источника вместо требуемого стиля.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния с полем answer.
    """
    notes = state["notes"]
    facts = "\n".join(f"- {fact}" for fact in notes.facts)

    llm = _build_output_llm().with_structured_output(Answer, method = "json_schema")

    answer = llm.invoke(
        [
            SystemMessage(content = COMPOSE_PROMPT),
            HumanMessage(
                content = (
                    f"Запрос пользователя:\n{state['question']}\n\n"
                    f"Сводка найденного:\n{notes.summary}\n\n"
                    f"Собранные факты:\n{facts}"
                )
            ),
        ]
    )

    print(f"[текст] разделов {len(answer.sections)}")
    return {"answer": answer}


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

    builder.add_node("agent", _agent_node)
    builder.add_node("tools", ToolNode(RESEARCH_TOOLS))
    builder.add_node("collect", _collect_node)
    builder.add_node("compose", _compose_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent, ["tools", "collect"])
    builder.add_edge("tools", "agent")
    builder.add_edge("collect", "compose")
    builder.add_edge("compose", END)

    return builder.compile()


def run_research(question: str) -> tuple[Answer, ResearchNotes]:
    """
    Прогоняет вопрос через граф.

    Аргументы:
        question: вопрос пользователя.

    Возвращает:
        Кортеж из итогового текста и фактической опоры, на которой он построен.
    """
    initial_state: ResearchState = {
        "question": question,
        "messages": [HumanMessage(content = question)],
        "search_rounds": 0,
        "notes": None,
        "answer": None,
    }

    # Запас по рекурсии: каждый раунд стоит два шага (agent + tools),
    # плюс финальный вызов agent и два узла вывода.
    final_state = build_graph().invoke(
        initial_state,
        config = {"recursion_limit": MAX_SEARCH_ROUNDS * 2 + 5},
    )

    return final_state["answer"], final_state["notes"]
