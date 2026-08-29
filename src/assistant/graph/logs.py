"""
Вывод хода прогона в консоль.
"""

from langchain_core.messages import AIMessage

from assistant.graph.budget import MAX_FAILED_CALLS, MAX_SUCCESSFUL_CALLS_PER_TOOL
from assistant.graph.state import Answer, ResearchNotes
from assistant.integrations.llm.client import reasoning_text


def log_round(round_number: int) -> None:
    """
    Печатает заголовок раунда.

    Аргументы:
        round_number: номер раунда, считая с единицы.

    Возвращает:
        Ничего.
    """
    print(f"\n--- раунд {round_number} ---")


def log_budget(
    successful_calls: dict[str, int],
    failed_calls: int,
    tools_left: list,
) -> None:
    """
    Печатает остаток бюджета и причину остановки, если она наступила.

    Аргументы:
        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.
        tools_left: инструменты, доступные на этом шаге.

    Возвращает:
        Ничего.
    """
    counters = ", ".join(
        f"{name} {count}/{MAX_SUCCESSFUL_CALLS_PER_TOOL}"
        for name, count in successful_calls.items()
    )
    print(f"[бюджет] {counters}, провалов {failed_calls}/{MAX_FAILED_CALLS}")

    if tools_left:
        return

    if failed_calls >= MAX_FAILED_CALLS:
        print("[бюджет] лимит неудачных вызовов исчерпан, отвечаю по собранному")
    else:
        print("[бюджет] все инструменты исчерпаны, отвечаю по собранному")


def log_decision(message: AIMessage) -> None:
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


def log_blocked_call(tool_name: str, reason: str) -> None:
    """
    Печатает отказ по бюджету на один вызов инструмента.

    Аргументы:
        tool_name: имя инструмента, вызов которого отклонён.
        reason: причина отказа.

    Возвращает:
        Ничего.
    """
    print(f"[бюджет] вызов {tool_name} отклонён: {reason}")


def log_notes(notes: ResearchNotes) -> None:
    """
    Печатает объём собранной фактической опоры.

    Аргументы:
        notes: фактическая опора, собранная узлом collect.

    Возвращает:
        Ничего.
    """
    print(f"\n[факты] собрано {len(notes.facts)}, источников {len(notes.sources)}")


def log_answer(answer: Answer) -> None:
    """
    Печатает объём итогового текста.

    Аргументы:
        answer: итоговый текст, собранный узлом compose.

    Возвращает:
        Ничего.
    """
    print(f"[текст] разделов {len(answer.sections)}")

def log_narrator_style(style: str) -> None:
    print(f"[СТИЛЬ]\n {style}")
