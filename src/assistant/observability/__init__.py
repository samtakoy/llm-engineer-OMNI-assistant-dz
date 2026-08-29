"""
Наблюдение за прогоном: показ на экране и журнал на диске.
"""

from .console import PACKAGE_LOGGER_NAME, setup_console_output

__all__ = [
    "setup_console_output",
    "PACKAGE_LOGGER_NAME",
]
