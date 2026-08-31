"""
Ход прогона в журнал.

Строки идут в logging под именем модуля; куда их показать, решает настройка
вывода в пакете observability.
"""

import logging

from langchain_core.messages import AIMessage

from assistant.graph.budget import MAX_FAILED_CALLS, MAX_SUCCESSFUL_CALLS_PER_TOOL
from assistant.graph.state import Answer, ResearchNotes

logger = logging.getLogger(__name__)

# Причины остановки, при которых ответ модели неполон.
INCOMPLETE_FINISH_REASONS = ("length", "content_filter")


def log_round(round_number: int) -> None:
    """
    Пишет заголовок раунда.

    Аргументы:
        round_number: номер раунда, считая с единицы.

    Возвращает:
        Ничего.
    """
    logger.info(f"\n--- раунд {round_number} ---")


def log_budget(
    successful_calls: dict[str, int],
    failed_calls: int,
    tools_left: list,
) -> None:
    """
    Пишет остаток бюджета и причину остановки, если она наступила.

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
    logger.info(f"[бюджет] {counters}, провалов {failed_calls}/{MAX_FAILED_CALLS}")

    if tools_left:
        return

    if failed_calls >= MAX_FAILED_CALLS:
        logger.info("[бюджет] лимит неудачных вызовов исчерпан, отвечаю по собранному")
    else:
        logger.info("[бюджет] все инструменты исчерпаны, отвечаю по собранному")


def log_decision(message: AIMessage) -> None:
    """
    Пишет решение модели на текущем шаге.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Ничего.
    """
    if message.tool_calls:
        for call in message.tool_calls:
            logger.info(f"[инструмент] {call['name']}({call['args']})")
    else:
        logger.info("[решение] инструменты больше не нужны, перехожу к сбору фактов")


def log_finish_reason(message: AIMessage) -> None:
    """
    Пишет причину остановки генерации, сообщённую сервером.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Ничего.
    """
    reason = message.response_metadata.get("finish_reason")

    if reason is None:
        logger.info("[генерация] причина остановки не сообщена")
        return

    if reason in INCOMPLETE_FINISH_REASONS:
        logger.warning(f"[генерация] ответ оборван, finish_reason: {reason}")
        return

    logger.info(f"[генерация] finish_reason: {reason}")


def log_blocked_call(tool_name: str, reason: str) -> None:
    """
    Пишет отказ по бюджету на один вызов инструмента.

    Аргументы:
        tool_name: имя инструмента, вызов которого отклонён.
        reason: причина отказа.

    Возвращает:
        Ничего.
    """
    logger.info(f"[бюджет] вызов {tool_name} отклонён: {reason}")


def log_notes(notes: ResearchNotes) -> None:
    """
    Пишет объём собранной фактической опоры.

    Аргументы:
        notes: фактическая опора, собранная узлом collect.

    Возвращает:
        Ничего.
    """
    logger.info(f"\n[факты] собрано {len(notes.facts)}, источников {len(notes.sources)}")


def log_answer(answer: Answer) -> None:
    """
    Пишет объём итогового текста.

    Аргументы:
        answer: итоговый текст, собранный узлом compose.

    Возвращает:
        Ничего.
    """
    logger.info(f"[текст] разделов {len(answer.sections)}")


def log_narrator_style(style: str) -> None:
    """
    Пишет фразу о голосе рассказчика.

    Аргументы:
        style: фраза, задающая манеру изложения.

    Возвращает:
        Ничего.
    """
    logger.info(f"[СТИЛЬ]\n {style}")


def log_run_id(run_id: str) -> None:
    """
    Пишет идентификатор прогона.

    Аргументы:
        run_id: идентификатор прогона; он же имя файла журнала и ключ снимков.

    Возвращает:
        Ничего.
    """
    logger.info(f"[прогон] {run_id}")


def log_resume(run_id: str, from_node: str) -> None:
    """
    Пишет, какой прогон и с какого узла переигрывается.

    Аргументы:
        run_id: идентификатор прогона.
        from_node: узел, с которого идёт продолжение.

    Возвращает:
        Ничего.
    """
    logger.info(f"[прогон] {run_id} продолжается с узла {from_node}")


def log_checkpoints_off() -> None:
    """
    Пишет, что снимки выключены и переиграть прогон будет нечем.

    Возвращает:
        Ничего.
    """
    logger.info("[снимки] выключены: переиграть этот прогон с середины не выйдет")
