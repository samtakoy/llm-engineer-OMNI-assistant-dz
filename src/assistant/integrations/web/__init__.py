"""
Веб-слой: поиск страниц, загрузка их текста и кеш того и другого.

Пакет самодостаточен и переносится в другой проект копированием папки. Он не
импортирует ничего из проекта, не читает окружение и не пишет в журнал: все
настройки приходят объектом WebConfig, а сообщения печатаются через print.

Внешние зависимости: httpx, trafilatura, ddgs.

Ни одна публичная функция не выбрасывает исключений наружу - при любой неудаче
возвращается пустой результат и печатается причина.

Пример использования описан в README.md рядом с этим файлом.
"""

from .config import WebConfig
from .outcomes import PageOutcome, SearchOutcome, SearchResult, ServiceFailure
from .pages import fetch_page
from .search import search

__all__ = [
    "WebConfig",
    "SearchResult",
    "ServiceFailure",
    "SearchOutcome",
    "PageOutcome",
    "search",
    "fetch_page",
]
