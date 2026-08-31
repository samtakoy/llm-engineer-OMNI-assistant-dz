"""
Утилиты графа: обработка текста, не связанная с логикой узлов.
"""

from .caps import CAPS_SHARE_THRESHOLD, build_case_reference, normalize_caps

__all__ = [
    "CAPS_SHARE_THRESHOLD",
    "build_case_reference",
    "normalize_caps",
]
