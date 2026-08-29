"""
Показ журнала прогона на экране.
"""

import logging
import sys

# Журнал пакета: обработчики вешаются на него, модули пишут в дочерние.
PACKAGE_LOGGER_NAME = "assistant"


def setup_console_output() -> None:
    """
    Направляет журнал пакета на стандартный вывод без служебных полей.

    Возвращает:
        Ничего. Повторный вызов обработчик не дублирует.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        return

    handler = logging.StreamHandler(stream = sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
