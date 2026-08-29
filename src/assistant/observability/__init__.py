"""
Наблюдение за прогоном: показ на экране и журнал прогона на диске.
"""

from .console import PACKAGE_LOGGER_NAME, setup_console_output
from .md_trace import MarkdownTrace, NoteHandler
from .tracing import build_callbacks

__all__ = [
    "setup_console_output",
    "build_callbacks",
    "MarkdownTrace",
    "NoteHandler",
    "PACKAGE_LOGGER_NAME",
]
