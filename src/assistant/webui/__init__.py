"""
Веб-точка входа поверх прогона ассистента.
"""

from .app import build_app, launch_app

__all__ = [
    "build_app",
    "launch_app",
]
