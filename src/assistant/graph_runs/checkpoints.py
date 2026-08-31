"""
Хранилище состояния графа: снимок после каждого шага.

Снимки лежат по ключу thread_id и позволяют продолжить прогон с любого узла.
Версия схемы входит в имя файла базы: состав полей ResearchState меняется -
версия растёт, и старые снимки не читаются вместо того, чтобы приехать
половиной ожидаемых полей.
"""

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


# Версия состава полей ResearchState.
CHECKPOINT_SCHEMA_VERSION = 1

# Наши схемы, которые лежат в снимке: без списка langgraph отказывается их
# восстанавливать в строгом режиме msgpack.
ALLOWED_STATE_TYPES = (
    ("assistant.graph.state", "ResearchNotes"),
    ("assistant.graph.state", "Answer"),
    ("assistant.graph.state", "Section"),
)


def open_checkpointer(directory: Path | None) -> BaseCheckpointSaver | None:
    """
    Открывает хранилище снимков, если оно включено.

    Аргументы:
        directory: каталог базы; None означает выключенное хранилище.

    Возвращает:
        Хранилище либо None, если оно выключено или библиотека недоступна.
        Исключений наружу не выбрасывает.
    """
    if directory is None:
        return None

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as error:
        logger.warning(f"[снимки] библиотека langgraph-checkpoint-sqlite недоступна: {error}")
        return None

    try:
        directory.mkdir(parents = True, exist_ok = True)
        path = directory / f"v{CHECKPOINT_SCHEMA_VERSION}.sqlite"
        # Соединение живёт весь прогон
        connection = sqlite3.connect(str(path), check_same_thread = False)
        # Список типов задаётся сериализатору напрямую: без него langgraph
        # восстанавливает из снимка что угодно и пишет об этом предупреждение.
        saver = SqliteSaver(
            connection,
            serde = JsonPlusSerializer(allowed_msgpack_modules = ALLOWED_STATE_TYPES),
        )
        saver.setup()
    except Exception as error:
        logger.warning(f"[снимки] хранилище не открылось: {type(error).__name__}: {error}")
        return None

    return saver
