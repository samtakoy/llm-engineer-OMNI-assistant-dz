"""
Подключение журнала прогона к графу.

Здесь живёт всё, что журнал знает о нашем графе: шапка файла, где взять запрос
пользователя и как свести итог. Сам журнал - кирпич без привязки к проекту.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from assistant.observability.console import PACKAGE_LOGGER_NAME
from assistant.observability.md_trace import MarkdownTrace, NoteHandler
from assistant.variables import LLM_PROVIDER, SHOW_REASONING, TRACE_DIR


def build_callbacks(node_rows: list[str]) -> list[BaseCallbackHandler]:
    """
    Собирает список слушателей для вызова графа.

    Аргументы:
        node_rows: описание моделей по узлам, строкой на узел; идёт в шапку.

    Возвращает:
        Список слушателей. Пустой, если журнал выключен.
    """
    if TRACE_DIR is None:
        return []

    return [_build_md_trace(directory = TRACE_DIR, node_rows = node_rows)]


def _build_md_trace(directory: Path, node_rows: list[str]) -> MarkdownTrace:
    """
    Заводит журнал на текущий прогон и подключает к нему logging.

    Аргументы:
        directory: каталог журналов; файл называется временем запуска.
        node_rows: описание моделей по узлам.

    Возвращает:
        Готовый слушатель.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace = MarkdownTrace(
        path = directory / f"{stamp}.md",
        header_rows = _header_rows(node_rows = node_rows),
        describe_request = _describe_request,
        summarize_result = _summarize_result,
    )

    # Строки узлов идут через logging и событиями LangChain не являются:
    # обработчик уводит их в тот же файл записями kind=note.
    handler = NoteHandler(trace = trace)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(handler)
    return trace


def _header_rows(node_rows: list[str]) -> list[str]:
    """
    Собирает шапку файла: на чём идёт прогон.

    Аргументы:
        node_rows: описание моделей по узлам.

    Возвращает:
        Строки для печати под заголовком.
    """
    return [
        f"- провайдер: `{LLM_PROVIDER}`",
        f"- размышление: {'включено' if SHOW_REASONING else 'выключено'}",
        "- модели по узлам:",
        *[f"    - {line}" for line in node_rows],
    ]


def _describe_request(inputs: Any) -> str:
    """
    Достаёт вопрос пользователя из стартового состояния.

    Аргументы:
        inputs: стартовое состояние графа.

    Возвращает:
        Текст вопроса либо пустую строку.
    """
    if not isinstance(inputs, dict):
        return ""

    return str(inputs.get("question") or "")


def _summarize_result(outputs: Any) -> str:
    """
    Сводит итоговое состояние графа в одну строку.

    Аргументы:
        outputs: итоговое состояние графа.

    Возвращает:
        Строку для спины журнала.
    """
    if not isinstance(outputs, dict):
        return "состояние не разобрано"

    notes = outputs.get("notes")
    answer = outputs.get("answer")

    if notes is None:
        return "факты не собраны"
    if answer is None:
        return f"факты собраны ({len(notes.facts)}), текст не сложен"

    return f"фактов {len(notes.facts)}, источников {len(notes.sources)}, разделов {len(answer.sections)}"
