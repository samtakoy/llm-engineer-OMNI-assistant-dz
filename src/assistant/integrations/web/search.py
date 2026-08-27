"""
Поиск страниц в интернете с кешем выдачи.

Функция не выбрасывает исключений наружу - при любой неудаче возвращается
пустой результат и печатается причина.

Кеш выдачи ценнее кеша страниц: с ним прогон повторяется целиком без сети.
Обратная сторона - выдача замораживается, и при работе над формулировками
поисковых запросов легко застрять на старой. Отсюда отдельный срок годности,
короче, чем у страниц, и обход кеша через настройку.
"""

import re

from .cache import FileCache, open_cache
from .config import WebConfig
from .outcomes import SearchOutcome, SearchResult, ServiceFailure

# Версия формата записи о выдаче. Двигается независимо от записи о странице.
_RECORD_VERSION = 1

_CACHE_NAMESPACE = "searches"

# Текст, которым ddgs сообщает о пустой выдаче, а не об отказе движка.
_EMPTY_RESULTS_MESSAGE = "No results found."

_REPEATED_SPACES = re.compile(r"\s+")


def search(query: str, max_results: int, config: WebConfig) -> SearchOutcome:
    """
    Выполняет поисковый запрос, спрашивая сначала кеш.

    Аргументы:
        query: поисковый запрос.
        max_results: сколько позиций выдачи вернуть.
        config: настройки веб-слоя.

    Возвращает:
        Исход поиска: найденные позиции и отказ поисковика, если он случился.
    """
    cache = open_cache(
        directory = config.cache_directory,
        namespace = _CACHE_NAMESPACE,
        ttl_days = config.search_cache_ttl_days,
        record_version = _RECORD_VERSION,
    )
    key = _cache_key(query = query)

    if cache is not None and not config.bypass_cache:
        results = _cached_results(cache = cache, key = key, max_results = max_results)
        if results is not None:
            print(f"[web] выдача по «{query}» взята из кеша: {len(results)} позиций")
            return SearchOutcome(results = results, failure = None)

    outcome = _search_online(query = query, max_results = max_results)

    # В кеш идёт только настоящая выдача. Отказ поисковика не кешируется -
    # ратлимит через час пройдёт, а запись о нём стоила бы живой выдачи.
    # Пустая выдача тоже: материал по запросу мог появиться, а промах здесь
    # стоит ровно одного обычного поиска.
    if cache is not None and outcome.failure is None and outcome.results:
        _store_results(cache = cache, key = key, max_results = max_results, results = outcome.results)

    return outcome


def _search_online(query: str, max_results: int) -> SearchOutcome:
    """
    Спрашивает поисковик.

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


def _cache_key(query: str) -> str:
    """
    Приводит запрос к ключу записи.

    Нормализация минимальная: обрезка краёв и сжатие повторных пробелов.
    Регистр не трогается - выигрыш в попаданиях копеечный, а разные запросы
    так можно склеить.

    Аргументы:
        query: поисковый запрос как его составила модель.

    Возвращает:
        Ключ записи в кеше.
    """
    return _REPEATED_SPACES.sub(" ", query.strip())


def _cached_results(cache: FileCache, key: str, max_results: int) -> list[SearchResult] | None:
    """
    Достаёт выдачу из кеша.

    Число запрошенных позиций в ключ не входит: иначе один запрос на пять и на
    десять позиций дал бы две записи. В записи лежит вся полученная тогда
    выдача, наружу отдаётся срез нужной длины.

    Попаданием считается запись, в которой позиций хватает, либо запись,
    сделанная по не меньшему запросу: если тогда просили десять, а поисковик
    отдал три, больше трёх у него и нет, и поход в сеть ничего не добавит.

    Аргументы:
        cache: хранилище выдачи.
        key: ключ записи.
        max_results: сколько позиций нужно сейчас.

    Возвращает:
        Позиции выдачи либо None, если годной записи нет или позиций мало.
    """
    record = cache.read(key = key)
    if record is None:
        return None

    stored_items = record.get("results")
    if not isinstance(stored_items, list) or not stored_items:
        return None

    requested_then = record.get("requested")
    has_enough = len(stored_items) >= max_results or (
        isinstance(requested_then, int) and requested_then >= max_results
    )
    if not has_enough:
        return None

    return [
        SearchResult(
            title = item.get("title", ""),
            url = item.get("url", ""),
            snippet = item.get("snippet", ""),
        )
        for item in stored_items[:max_results]
        if isinstance(item, dict) and item.get("url")
    ]


def _store_results(
    cache: FileCache,
    key: str,
    max_results: int,
    results: list[SearchResult],
) -> None:
    """
    Кладёт выдачу в кеш.

    Аргументы:
        cache: хранилище выдачи.
        key: ключ записи.
        max_results: сколько позиций просили в этот раз.
        results: полученные позиции выдачи.

    Возвращает:
        Ничего.
    """
    cache.write(
        key = key,
        payload = {
            "requested": max_results,
            "results": [
                {"title": item.title, "url": item.url, "snippet": item.snippet}
                for item in results
            ],
        },
    )
