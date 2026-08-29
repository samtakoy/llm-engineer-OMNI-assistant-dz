"""
Учёт бюджета инструментов по истории сообщений.

Счёт идёт вызовами, а не раундами. У каждого инструмента свой лимит
состоявшихся вызовов, у провалов - общий лимит на прогон.
"""

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool

from assistant.graph.prompts import TOOL_BUDGET_NOTE
from assistant.graph.tools import CALL_BLOCKED, CALL_COMPLETED, RESEARCH_TOOLS

# Сколько состоявшихся вызовов разрешено каждому инструменту.
MAX_SUCCESSFUL_CALLS_PER_TOOL = 6

# Сколько вызовов может провалиться на весь прогон. Провалы не тратят бюджет
# инструмента.
MAX_FAILED_CALLS = 6

# Потолок вызовов за прогон, из него считается запас по рекурсии.
MAX_TOOL_CALLS_PER_RUN = MAX_SUCCESSFUL_CALLS_PER_TOOL * len(RESEARCH_TOOLS) + MAX_FAILED_CALLS


def count_tool_calls(messages: list[AnyMessage]) -> tuple[dict[str, int], int]:
    """
    Считает бюджет инструментов по истории диалога.

    Правило счёта по полю artifact: CALL_BLOCKED - не считается,
    CALL_COMPLETED - в бюджет своего инструмента, остальное - в провалы.

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


def available_tools(successful_calls: dict[str, int], failed_calls: int) -> list[BaseTool]:
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


def budget_note(
    successful_calls: dict[str, int],
    failed_calls: int,
    tools_left: list[BaseTool],
) -> str:
    """
    Составляет заметку о бюджете для системного сообщения.

    Аргументы:
        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.
        tools_left: инструменты, доступные на этом шаге.

    Возвращает:
        Текст заметки.
    """
    available_names = {tool.name for tool in tools_left}

    lines = []
    for tool in RESEARCH_TOOLS:
        remaining = MAX_SUCCESSFUL_CALLS_PER_TOOL - successful_calls[tool.name]
        state = f"доступен, осталось вызовов: {remaining}" if tool.name in available_names else "исчерпан"
        lines.append(f"- {tool.name}: {state}")

    lines.append(f"- неудачных вызовов: {failed_calls} из {MAX_FAILED_CALLS}")

    return TOOL_BUDGET_NOTE.format(budget_lines = "\n".join(lines))
