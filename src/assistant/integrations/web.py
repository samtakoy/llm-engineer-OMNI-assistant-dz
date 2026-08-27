"""
Веб-слой ресёрчера: поиск и извлечение текста страниц.

Ни одна функция здесь не выбрасывает исключений наружу — при любой неудаче
возвращается пустой результат и печатается причина. Ресёрчер должен переживать
недоступный сайт, ратлимит поисковика и отсутствие сети, а не ронять прогон.

Неудачи различаются по двум осям. Первая: отказ сервиса против пустого ответа -
по ней граф ведёт бюджет вызовов. Вторая: временный отказ против постоянного -
по ней ресёрчер решает, повторять вызов или отказаться от этого пути. Обе оси
вместе с краткой причиной уходят в диалог: без них модель повторяет мёртвый
адрес и бросает запрос, который заработал бы со второй попытки.
"""

import ipaddress
import socket
import ssl
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura

_TIMEOUT_SECONDS = 10.0
_MAX_BYTES = 2_000_000
_USER_AGENT = "smm-department-bot/0.1 (+learning project)"

# Текст, которым ddgs сообщает о пустой выдаче, а не об отказе движка.
_EMPTY_RESULTS_MESSAGE = "No results found."

# Коды ответа, после которых повтор того же запроса ничего не изменит: сервер
# понял запрос и отказал. Всё остальное, включая 429 и пятисотые, считается
# временным.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 410, 451})


@dataclass(frozen = True)
class SearchResult:
    """Одна позиция поисковой выдачи."""

    title: str
    url: str
    snippet: str


@dataclass(frozen = True)
class ServiceFailure:
    """
    Отказ сервиса.

    Поля:
        reason: краткая причина одной строкой, уходит в диалог с моделью.
        is_temporary: повтор того же вызова имеет смысл.
    """

    reason: str
    is_temporary: bool


@dataclass(frozen = True)
class SearchOutcome:
    """
    Исход поискового запроса.

    Пустая выдача и отказ сервиса разделены намеренно: пустая выдача - это
    ответ, по которому ресёрчер меняет формулировку, а отказ сервиса ответом
    не является и должен считаться отдельно.

    Поля:
        results: найденные позиции выдачи.
        failure: отказ поисковика, None если поисковик отработал.
    """

    results: list[SearchResult]
    failure: ServiceFailure | None


@dataclass(frozen = True)
class PageOutcome:
    """
    Исход загрузки страницы.

    Поля:
        text: основной текст страницы, пустая строка при любой неудаче.
        failure: отказ сайта, None если ответ получен.
        empty_reason: почему текста нет, когда отказа не было. Пустая строка,
            если текст получен.
    """

    text: str
    failure: ServiceFailure | None
    empty_reason: str


def search(query: str, max_results: int) -> SearchOutcome:
    """
    Выполняет поисковый запрос.

    Аргументы:
        query: поисковый запрос.
        max_results: сколько позиций выдачи вернуть.

    Возвращает:
        Исход поиска: найденные позиции и отказ поисковика, если он случился.
    """
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    except ImportError as error:
        print(f"[web] библиотека ddgs недоступна: {error}")
        return SearchOutcome(
            results = [],
            failure = ServiceFailure(
                reason = "библиотека поиска не установлена",
                is_temporary = False,
            ),
        )

    try:
        with DDGS() as ddgs:
            items = list(ddgs.text(query, max_results = max_results))
    except DDGSException as error:
        if _is_empty_results_error(error = error):
            print(f"[web] по запросу «{query}» ничего не нашлось")
            return SearchOutcome(results = [], failure = None)

        print(f"[web] поиск «{query}» не удался: {type(error).__name__}: {error}")

        if isinstance(error, TimeoutException):
            reason = "поисковик не ответил вовремя"
        elif isinstance(error, RatelimitException):
            reason = "поисковик ограничил частоту запросов"
        else:
            reason = "поисковик не отдал выдачу"

        return SearchOutcome(
            results = [],
            failure = ServiceFailure(reason = reason, is_temporary = True),
        )
    except Exception as error:
        print(f"[web] поиск «{query}» не удался: {type(error).__name__}: {error}")
        return SearchOutcome(
            results = [],
            failure = ServiceFailure(
                reason = "поиск оборвался с ошибкой",
                is_temporary = True,
            ),
        )

    results = [
        SearchResult(
            title = item.get("title", ""),
            url = item.get("href", ""),
            snippet = item.get("body", ""),
        )
        for item in items
        if item.get("href")
    ]

    return SearchOutcome(results = results, failure = None)


def _is_empty_results_error(error: Exception) -> bool:
    """
    Отличает пустую выдачу от отказа поисковика.

    Библиотека ddgs сообщает об обоих случаях одним исключением: у пустой
    выдачи текст ровно _EMPTY_RESULTS_MESSAGE, у отказа - текст упавшего
    движка, у ратлимита и таймаута - отдельные подклассы. Опора на текст
    хрупкая: при смене формулировки в ddgs пустая выдача начнёт считаться
    отказом.

    Аргументы:
        error: исключение, пришедшее из поиска.

    Возвращает:
        True, если поисковик отработал и ничего не нашёл.
    """
    from ddgs.exceptions import DDGSException

    return type(error) is DDGSException and str(error) == _EMPTY_RESULTS_MESSAGE


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


def _has_certificate_cause(error: Exception) -> bool:
    """
    Ищет в цепочке причин исключения ошибку проверки сертификата.

    Проверяется тип, а не текст сообщения: формулировка ошибки ssl меняется от
    версии python к версии. Цепочка обходится и по __cause__, и по __context__:
    httpx связывает своё исключение с httpcore явно, а httpcore с ошибкой ssl -
    неявно, только через контекст.

    Аргументы:
        error: исключение, пришедшее из httpx.

    Возвращает:
        True, если причина отказа - сертификат сайта.
    """
    seen = set()
    current = error

    while current is not None and id(current) not in seen:
        if isinstance(current, ssl.SSLError):
            return True

        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return False


def _describe_http_failure(error: httpx.HTTPError) -> ServiceFailure:
    """
    Переводит ошибку httpx в причину, понятную модели.

    Аргументы:
        error: исключение, пришедшее из httpx.

    Возвращает:
        Отказ сайта с краткой причиной и признаком временности.
    """
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return ServiceFailure(
            reason = f"сервер ответил кодом {status_code}",
            is_temporary = status_code not in _PERMANENT_STATUS_CODES,
        )

    if _has_certificate_cause(error = error):
        return ServiceFailure(
            reason = "сертификат сайта не проходит проверку",
            is_temporary = False,
        )

    if isinstance(error, httpx.TimeoutException):
        return ServiceFailure(reason = "сайт не ответил вовремя", is_temporary = True)

    return ServiceFailure(reason = "соединение с сайтом не установилось", is_temporary = True)


def fetch_page(url: str, max_characters: int) -> PageOutcome:
    """
    Скачивает страницу и извлекает из неё основной текст.

    Отказом считается только неответ сайта. Отбракованный адрес, чужой тип
    содержимого и страница без текста - это полученный ответ, по которому
    ресёрчер берёт следующий адрес.

    Аргументы:
        url: адрес страницы.
        max_characters: до скольких символов обрезать результат.

    Возвращает:
        Исход загрузки: текст статьи, отказ сайта и причина пустого результата.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        print(f"[web] пропуск {url}: схема {parsed.scheme!r} не поддерживается")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = f"схема {parsed.scheme!r} не поддерживается",
        )
    if not parsed.hostname:
        print(f"[web] пропуск {url}: в адресе нет хоста")
        return PageOutcome(text = "", failure = None, empty_reason = "в адресе нет хоста")
    if _is_blocked_host(parsed.hostname):
        print(f"[web] пропуск {url}: хост ведёт во внутреннюю сеть")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = "хост не резолвится или ведёт во внутреннюю сеть",
        )

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
                    return PageOutcome(
                        text = "",
                        failure = None,
                        empty_reason = f"тип содержимого {content_type!r}, а не html",
                    )

                chunks, downloaded_bytes = [], 0
                for chunk in response.iter_bytes():
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > _MAX_BYTES:
                        break
                    chunks.append(chunk)

                html = b"".join(chunks).decode(response.encoding or "utf-8", errors = "replace")
    except httpx.HTTPError as error:
        print(f"[web] не удалось скачать {url}: {type(error).__name__}: {error}")
        return PageOutcome(
            text = "",
            failure = _describe_http_failure(error = error),
            empty_reason = "",
        )

    text = trafilatura.extract(html, include_comments = False, include_tables = False)
    if not text:
        print(f"[web] из {url} не извлёкся текст")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = "на странице нет связного текста статьи",
        )

    text = text.strip()
    if len(text) > max_characters:
        text = text[:max_characters] + "\n…[обрезано]"

    return PageOutcome(text = text, failure = None, empty_reason = "")
