"""
Загрузка страниц и извлечение из них основного текста, с кешем текста.

Функция не выбрасывает исключений наружу - при любой неудаче возвращается
пустой текст и печатается причина. Вызывающий код должен переживать
недоступный сайт и отсутствие сети, а не ронять прогон.

В кеш кладётся полный текст страницы, обрезка по длине делается уже после
чтения: иначе смена лимита требовала бы перекачки всего собранного.
"""

import ipaddress
import logging
import socket
import ssl
from urllib.parse import urlparse

import httpx
import trafilatura

from ..filecache import FileCache, open_cache
from .config import WebConfig
from .outcomes import PageOutcome, ServiceFailure

logger = logging.getLogger(__name__)


# Версия формата записи о странице. Двигается независимо от записи о выдаче.
_RECORD_VERSION = 1

_CACHE_NAMESPACE = "pages"

# Коды ответа, после которых повтор того же запроса ничего не изменит: сервер
# понял запрос и отказал. Всё остальное, включая 429 и пятисотые, считается
# временным.
_PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 410, 451})


def fetch_page(url: str, max_characters: int, config: WebConfig) -> PageOutcome:
    """
    Отдаёт основной текст страницы, спрашивая сначала кеш.

    Порядок проверок: схема, кеш, хост, сеть. Кеш стоит до проверки хоста
    намеренно: сетевого запроса при попадании не делается, текст добыт с
    публичного адреса раньше, и попадание не платит за резолв имени.

    Отказом считается только неответ сайта. Отбракованный адрес, чужой тип
    содержимого и страница без текста - это полученный ответ, по которому
    вызывающий код берёт следующий адрес.

    Аргументы:
        url: адрес страницы.
        max_characters: до скольких символов обрезать результат.
        config: настройки веб-слоя.

    Возвращает:
        Исход загрузки: текст статьи, отказ сайта и причина пустого результата.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        logger.warning(f"[web] пропуск {url}: схема {parsed.scheme!r} не поддерживается")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = f"схема {parsed.scheme!r} не поддерживается",
        )
    if not parsed.hostname:
        logger.warning(f"[web] пропуск {url}: в адресе нет хоста")
        return PageOutcome(text = "", failure = None, empty_reason = "в адресе нет хоста")

    cache = open_cache(
        directory = config.cache_directory,
        namespace = _CACHE_NAMESPACE,
        ttl_days = config.page_cache_ttl_days,
        record_version = _RECORD_VERSION,
    )

    if cache is not None and not config.bypass_cache:
        cached_text = _cached_text(cache = cache, url = url)
        if cached_text:
            logger.info(f"[web] текст {url} взят из кеша: {len(cached_text)} символов")
            return PageOutcome(
                text = _truncated(text = cached_text, max_characters = max_characters),
                failure = None,
                empty_reason = "",
            )

    if _is_blocked_host(host = parsed.hostname):
        logger.warning(f"[web] пропуск {url}: хост ведёт во внутреннюю сеть")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = "хост не резолвится или ведёт во внутреннюю сеть",
        )

    outcome = _fetch_online(url = url, config = config)
    if outcome.text and cache is not None:
        cache.write(key = url, payload = {"characters": len(outcome.text), "text": outcome.text})

    if not outcome.text:
        return outcome

    return PageOutcome(
        text = _truncated(text = outcome.text, max_characters = max_characters),
        failure = None,
        empty_reason = "",
    )


def _fetch_online(url: str, config: WebConfig) -> PageOutcome:
    """
    Скачивает страницу и извлекает из неё основной текст целиком, без обрезки.

    Аргументы:
        url: адрес страницы.
        config: настройки веб-слоя.

    Возвращает:
        Исход загрузки: полный текст статьи, отказ сайта и причина пустого
        результата.
    """
    try:
        with httpx.Client(
            timeout = config.request_timeout_seconds,
            follow_redirects = True,
            headers = {"User-Agent": config.user_agent},
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    logger.warning(f"[web] пропуск {url}: тип содержимого {content_type!r}")
                    return PageOutcome(
                        text = "",
                        failure = None,
                        empty_reason = f"тип содержимого {content_type!r}, а не html",
                    )

                chunks, downloaded_bytes = [], 0
                for chunk in response.iter_bytes():
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > config.max_page_bytes:
                        break
                    chunks.append(chunk)

                html = b"".join(chunks).decode(response.encoding or "utf-8", errors = "replace")
    except httpx.HTTPError as error:
        logger.warning(f"[web] не удалось скачать {url}: {type(error).__name__}: {error}")
        return PageOutcome(
            text = "",
            failure = _describe_http_failure(error = error),
            empty_reason = "",
        )

    text = trafilatura.extract(html, include_comments = False, include_tables = False)
    if not text:
        logger.warning(f"[web] из {url} не извлёкся текст")
        return PageOutcome(
            text = "",
            failure = None,
            empty_reason = "на странице нет связного текста статьи",
        )

    return PageOutcome(text = text.strip(), failure = None, empty_reason = "")


def _cached_text(cache: FileCache, url: str) -> str:
    """
    Достаёт полный текст страницы из кеша.

    Аргументы:
        cache: хранилище страниц.
        url: адрес страницы.

    Возвращает:
        Текст страницы либо пустую строку, если годной записи нет.
    """
    record = cache.read(key = url)
    if record is None:
        return ""

    text = record.get("text")
    if not isinstance(text, str):
        return ""

    return text


def _truncated(text: str, max_characters: int) -> str:
    """
    Обрезает текст до нужной длины, помечая обрезку.

    Аргументы:
        text: полный текст страницы.
        max_characters: до скольких символов обрезать.

    Возвращает:
        Текст не длиннее лимита, с пометкой в конце, если обрезка случилась.
    """
    if len(text) <= max_characters:
        return text

    return text[:max_characters] + "\n…[обрезано]"


def _is_blocked_host(host: str) -> bool:
    """
    Проверяет, не ведёт ли адрес во внутреннюю сеть.

    Защита от SSRF: поисковая выдача - внешние данные, и по ней нельзя
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
