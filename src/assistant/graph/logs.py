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

        round_number: номер раунда, считая с единицы.
    """
    logger.info(f"\n--- раунд {round_number} ---")


def log_budget(
    successful_calls: dict[str, int],
    failed_calls: int,
    tools_left: list,
) -> None:
    """
    Пишет остаток бюджета и причину остановки, если она наступила.

        successful_calls: сколько состоявшихся вызовов у каждого инструмента.
        failed_calls: сколько вызовов провалилось всего.
        tools_left: инструменты, доступные на этом шаге.
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

        message: ответ модели.
    """
    if message.tool_calls:
        for call in message.tool_calls:
            logger.info(f"[инструмент] {call['name']}({call['args']})")
    else:
        logger.info("[решение] инструменты больше не нужны, перехожу к сбору фактов")


def log_finish_reason(message: AIMessage) -> None:
    """
    Пишет причину остановки генерации, сообщённую сервером.

        message: ответ модели.
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

        tool_name: имя инструмента, вызов которого отклонён.
        reason: причина отказа.
    """
    logger.info(f"[бюджет] вызов {tool_name} отклонён: {reason}")


def log_notes(notes: ResearchNotes) -> None:
    """
    Пишет объём собранной фактической опоры.

        notes: фактическая опора, собранная узлом collect.
    """
    logger.info(f"\n[факты] собрано {len(notes.facts)}, источников {len(notes.sources)}")


def log_answer(answer: Answer) -> None:
    """
    Пишет объём итогового текста.

        answer: итоговый текст, собранный узлом compose.
    """
    logger.info(f"[текст] разделов {len(answer.sections)}")


def log_narrator_style(style: str) -> None:
    """
    Пишет фразу о голосе рассказчика.

        style: фраза, задающая манеру изложения.
    """
    logger.info(f"[СТИЛЬ]\n {style}")
