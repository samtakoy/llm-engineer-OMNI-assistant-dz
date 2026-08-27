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

Бюджет инструментов считается не раундами, а вызовами, и считается из истории
сообщений. У каждого инструмента свой лимит состоявшихся вызовов, у провалов -
общий лимит на прогон. Исчерпавший лимит инструмент перестаёт показываться
модели.
"""

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from assistant.graph.prompts import (
    COLLECT_PROMPT,
    COMPOSE_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    TOOL_BUDGET_NOTE,
)
from assistant.graph.state import Answer, ResearchNotes, ResearchState
from assistant.graph.tools import (
    CALL_BLOCKED,
    CALL_COMPLETED,
    RESEARCH_TOOLS,
)
from assistant.integrations.llm.client import build_llm, describe_llm, reasoning_text
from assistant.variables import SHOW_REASONING

# Сколько состоявшихся вызовов разрешено каждому инструменту. Лимит один на
# все инструменты: список инструментов растёт, а правило остаётся прежним.
MAX_SUCCESSFUL_CALLS_PER_TOOL = 6

# Сколько вызовов может провалиться на весь прогон. Отдельный бюджет нужен,
# потому что провалы не тратят бюджет инструмента: иначе один недоступный
# поисковик съедал бы всю квоту поиска, а без потолка прогон крутился бы на
# отказах сети до упора по рекурсии.
MAX_FAILED_CALLS = 6

# Исполнитель инструментов. Бюджет проверяется до него, в _tools_node.
_TOOL_EXECUTOR = ToolNode(RESEARCH_TOOLS)

# Потолок вызовов за прогон, из него считается запас по рекурсии.
_MAX_TOOL_CALLS = MAX_SUCCESSFUL_CALLS_PER_TOOL * len(RESEARCH_TOOLS) + MAX_FAILED_CALLS

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


def _count_tool_calls(messages: list[AnyMessage]) -> tuple[dict[str, int], int]:
    """
    Считает бюджет инструментов по истории диалога.

    Исход вызова помечает сам инструмент полем artifact. Отклонённые по бюджету
    вызовы не тратят ничего, выполненные тратят бюджет своего инструмента,
    остальное считается провалом - включая вызов несуществующего инструмента,
    на который ToolNode отвечает ошибкой без artifact.

    Аргументы:
        messages: история диалога.

    Возвращает:
        Кортеж: сколько состоявшихся вызовов у каждого инструмента и сколько
        вызовов провалилось всего.
    """
    successful_calls = {tool.name: 0 for tool in RESEARCH_TOOLS}
    failed_calls = 0

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        if message.artifact == CALL_BLOCKED:
            continue

        if message.artifact == CALL_COMPLETED and message.name in successful_calls:
            successful_calls[message.name] += 1
        else:
            failed_calls += 1

    return successful_calls, failed_calls


def _available_tools(successful_calls: dict[str, int], failed_calls: int) -> list[BaseTool]:
    """
    Отбирает инструменты, которые ещё можно показать модели.

    Аргументы:
        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.

    Возвращает:
        Список инструментов. Пустой, если бюджет провалов исчерпан или
        исчерпаны все инструменты.
    """
    if failed_calls >= MAX_FAILED_CALLS:
        return []

    return [
        tool
        for tool in RESEARCH_TOOLS
        if successful_calls[tool.name] < MAX_SUCCESSFUL_CALLS_PER_TOOL
    ]


def _budget_note(
    successful_calls: dict[str, int],
    failed_calls: int,
    available_tools: list[BaseTool],
) -> str:
    """
    Составляет заметку о бюджете для системного сообщения.

    Модель видит исчерпанные инструменты в своей же истории и без заметки
    пробует вызвать их снова, тратя раунд на ошибку.

    Аргументы:
        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.
        available_tools: инструменты, доступные на этом шаге.

    Возвращает:
        Текст заметки.
    """
    available_names = {tool.name for tool in available_tools}

    lines = []
    for tool in RESEARCH_TOOLS:
        remaining = MAX_SUCCESSFUL_CALLS_PER_TOOL - successful_calls[tool.name]
        state = f"доступен, осталось вызовов: {remaining}" if tool.name in available_names else "исчерпан"
        lines.append(f"- {tool.name}: {state}")

    lines.append(f"- неудачных вызовов: {failed_calls} из {MAX_FAILED_CALLS}")

    return TOOL_BUDGET_NOTE.format(budget_lines = "\n".join(lines))


def _log_budget(
    successful_calls: dict[str, int],
    failed_calls: int,
    available_tools: list[BaseTool],
) -> None:
    """
    Печатает остаток бюджета и причину остановки, если она наступила.

    Аргументы:
        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.
        available_tools: инструменты, доступные на этом шаге.

    Возвращает:
        Ничего.
    """
    counters = ", ".join(
        f"{name} {count}/{MAX_SUCCESSFUL_CALLS_PER_TOOL}"
        for name, count in successful_calls.items()
    )
    print(f"[бюджет] {counters}, провалов {failed_calls}/{MAX_FAILED_CALLS}")

    if available_tools:
        return

    if failed_calls >= MAX_FAILED_CALLS:
        print("[бюджет] лимит неудачных вызовов исчерпан, отвечаю по собранному")
    else:
        print("[бюджет] все инструменты исчерпаны, отвечаю по собранному")


def _log_decision(message: AIMessage) -> None:
    """
    Печатает размышление модели и её решение на текущем шаге.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Ничего.
    """
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

    Исчерпавший бюджет инструмент модели не показывается: она физически не
    может его вызвать. Когда доступных инструментов не осталось, вызов идёт без
    них, и граф гарантированно доходит до ответа, не оставляя вызовов без
    ответа.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния: ответ модели.
    """
    successful_calls, failed_calls = _count_tool_calls(messages = state["messages"])
    available_tools = _available_tools(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
    )

    round_number = sum(1 for message in state["messages"] if isinstance(message, AIMessage)) + 1
    print(f"\n--- раунд {round_number} ---")
    _log_budget(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
        available_tools = available_tools,
    )

    llm = _build_agent_llm()
    if available_tools:
        llm = llm.bind_tools(available_tools)

    budget_note = _budget_note(
        successful_calls = successful_calls,
        failed_calls = failed_calls,
        available_tools = available_tools,
    )

    message = llm.invoke(
        [
            SystemMessage(content = RESEARCHER_SYSTEM_PROMPT),
            SystemMessage(content = budget_note),
            *state["messages"],
        ]
    )

    _log_decision(message = message)

    return {"messages": [message]}


def _tools_node(state: ResearchState) -> dict:
    """
    Выполняет вызовы инструментов, отсекая те, на которые не осталось бюджета.

    Проверка нужна отдельно от отбора инструментов в узле agent: модель видит в
    истории исчерпанный инструмент и может вызвать его по образцу, а ToolNode
    выполнит любой известный ему вызов. Бюджет тратится по одному вызову, так
    что и пачка параллельных вызовов не может его перебрать.

    Ответ возвращается на каждый вызов без исключений: вызов без ответа ломает
    следующий запрос к модели.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Обновление состояния: по одному сообщению на каждый вызов.
    """
    successful_calls, failed_calls = _count_tool_calls(messages = state["messages"])
    remaining_calls = {
        name: MAX_SUCCESSFUL_CALLS_PER_TOOL - count
        for name, count in successful_calls.items()
    }
    budget_left = failed_calls < MAX_FAILED_CALLS

    last_message = state["messages"][-1]
    allowed_calls, blocked_messages = [], []

    for call in last_message.tool_calls:
        # Неизвестное имя пропускается к исполнителю намеренно: он ответит
        # ошибкой, и вызов зачтётся в провалы, а не в отказы по бюджету.
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
        print(f"[бюджет] вызов {call['name']} отклонён: {reason}")
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

    # Порядок ответов возвращается к порядку вызовов: так лог и история читаются
    # рядом с тем, что просила модель.
    by_call_id = {
        message.tool_call_id: message
        for message in [*executed_messages, *blocked_messages]
    }
    return {"messages": [by_call_id[call["id"]] for call in last_message.tool_calls]}


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
    builder.add_node("tools", _tools_node)
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
        "notes": None,
        "answer": None,
    }

    # Запас по рекурсии: каждый раунд тратит хотя бы один вызов инструмента и
    # стоит два шага (agent + tools), плюс финальный вызов agent и два узла
    # вывода. Бюджеты вызовов делают прогон конечным сами по себе.
    final_state = build_graph().invoke(
        initial_state,
        config = {"recursion_limit": _MAX_TOOL_CALLS * 2 + 5},
    )

    return final_state["answer"], final_state["notes"]
