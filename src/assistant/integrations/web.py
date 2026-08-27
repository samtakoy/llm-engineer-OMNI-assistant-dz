"""
Веб-слой ресёрчера: поиск и извлечение текста страниц.

Ни одна функция здесь не выбрасывает исключений наружу — при любой неудаче
возвращается пустой результат и печатается причина. Ресёрчер должен переживать
недоступный сайт, ратлимит поисковика и отсутствие сети, а не ронять прогон.
"""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura

_TIMEOUT_SECONDS = 10.0
_MAX_BYTES = 2_000_000
_USER_AGENT = "smm-department-bot/0.1 (+learning project)"


@dataclass(frozen = True)
class SearchResult:
    """Одна позиция поисковой выдачи."""

    title: str
    url: str
    snippet: str


def search(query: str, max_results: int) -> list[SearchResult]:
    """
    Выполняет поисковый запрос.

    Аргументы:
        query: поисковый запрос.
        max_results: сколько позиций выдачи вернуть.

    Возвращает:
        Список результатов. Пустой список, если поиск не удался или ничего
        не нашёл.
    """
    try:
        from ddgs import DDGS
    except ImportError as error:
        print(f"[web] библиотека ddgs недоступна: {error}")
        return []

    try:
        with DDGS() as ddgs:
            items = list(ddgs.text(query, max_results = max_results))
    except Exception as error:
        print(f"[web] поиск «{query}» не удался: {type(error).__name__}: {error}")
        return []

    return [
        SearchResult(
            title = item.get("title", ""),
            url = item.get("href", ""),
            snippet = item.get("body", ""),
        )
        for item in items
        if item.get("href")
    ]


def _is_blocked_host(host: str) -> bool:
    """
    Проверяет, не ведёт ли адрес во внутреннюю сеть.

    Защита от SSRF: поисковая выдача — внешние данные, и по ней нельзя
    ходить на localhost или в приватные подсети.

    Аргументы:
        host: имя хоста из URL.

    Возвращает:
        True, если хост не резолвится или резолвится в приватный адрес.
    """
    try:
        address_infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True

    for info in address_infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return True

    return False


def fetch_page(url: str, max_characters: int) -> str:
    """
    Скачивает страницу и извлекает из неё основной текст.

    Аргументы:
        url: адрес страницы.
        max_characters: до скольких символов обрезать результат.

    Возвращает:
        Очищенный текст статьи. Пустая строка, если страница недоступна,
        не является HTML или текст из неё извлечь не удалось.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        print(f"[web] пропуск {url}: схема {parsed.scheme!r} не поддерживается")
        return ""
    if not parsed.hostname:
        print(f"[web] пропуск {url}: в адресе нет хоста")
        return ""
    if _is_blocked_host(parsed.hostname):
        print(f"[web] пропуск {url}: хост ведёт во внутреннюю сеть")
        return ""

    try:
        with httpx.Client(
            timeout = _TIMEOUT_SECONDS,
            follow_redirects = True,
            headers = {"User-Agent": _USER_AGENT},
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    print(f"[web] пропуск {url}: тип содержимого {content_type!r}")
                    return ""

                chunks, downloaded_bytes = [], 0
                for chunk in response.iter_bytes():
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > _MAX_BYTES:
                        break
                    chunks.append(chunk)

                html = b"".join(chunks).decode(response.encoding or "utf-8", errors = "replace")
    except httpx.HTTPError as error:
        print(f"[web] не удалось скачать {url}: {type(error).__name__}: {error}")
        return ""

    text = trafilatura.extract(html, include_comments = False, include_tables = False)
    if not text:
        print(f"[web] из {url} не извлёкся текст")
        return ""

    text = text.strip()
    if len(text) > max_characters:
        text = text[:max_characters] + "\n…[обрезано]"

    return text
