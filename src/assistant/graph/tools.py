"""
Инструменты ресёрчера: поиск и загрузка страницы.

Тонкие обёртки над integrations.web. Их задача - превратить результат в текст,
пригодный для модели, и никогда не бросать исключение: неудача инструмента
должна вернуться в диалог сообщением, а не уронить граф.
"""

from langchain_core.tools import tool

from assistant.integrations.web import fetch_page, search

_MAX_SEARCH_RESULTS = 5
_MAX_PAGE_CHARACTERS = 4000


@tool
def search_web(query: str) -> str:
    """Ищет страницы в интернете по запросу. Возвращает заголовок, адрес и краткое
    описание каждой найденной страницы."""
    results = search(query = query, max_results = _MAX_SEARCH_RESULTS)

    if not results:
        return f"По запросу «{query}» ничего не найдено. Попробуй другую формулировку."

    return "\n\n".join(
        f"{position}. {item.title}\n{item.url}\n{item.snippet}"
        for position, item in enumerate(results, start = 1)
    )


@tool
def fetch_url(url: str) -> str:
    """Скачивает страницу по адресу и возвращает её основной текст. Адрес брать
    только из выдачи search_web."""
    text = fetch_page(url = url, max_characters = _MAX_PAGE_CHARACTERS)

    if not text:
        return f"Страницу {url} прочитать не удалось. Возьми другой адрес из выдачи."

    return f"Текст страницы {url}:\n\n{text}"


RESEARCH_TOOLS = [search_web, fetch_url]
