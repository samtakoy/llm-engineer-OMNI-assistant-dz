"""
Инструменты ресёрчера: поиск и загрузка страницы.

Тонкие обёртки над integrations.web. Их задача - превратить результат в текст,
пригодный для модели, и никогда не бросать исключение: неудача инструмента
должна вернуться в диалог сообщением, а не уронить граф.

Каждый инструмент возвращает пару: текст для модели и исход вызова. Исход
уходит в поле artifact сообщения ToolMessage, модель его не видит, а граф по
нему ведёт бюджет вызовов. Выполненным считается вызов, на который сервис
ответил, даже если ответ пуст: пустая выдача - это повод сменить формулировку,
а не сбой.

В текст неудачи всегда входит причина и совет, повторять вызов или нет. Причина
без совета бесполезна: модель видит «сбой» и решает наугад.
"""

from langchain_core.tools import tool

from assistant.integrations.web import ServiceFailure, fetch_page, search

_MAX_SEARCH_RESULTS = 5
_MAX_PAGE_CHARACTERS = 4000

# Исходы вызова, по которым граф ведёт бюджет. Трёх значений хватает: два
# тратят бюджет по-разному, третье не тратит ничего.
CALL_COMPLETED = "выполнен"
CALL_FAILED = "провален"
CALL_BLOCKED = "отклонён"


def _advice_on_failure(failure: ServiceFailure, repeat_hint: str) -> str:
    """
    Составляет совет по отказу сервиса.

    Без совета модель ведёт себя ровно наоборот нужного: бросает запрос,
    который заработал бы со второй попытки, и упрямо повторяет тот, который
    не заработает никогда.

    Аргументы:
        failure: отказ сервиса.
        repeat_hint: как именно повторить вызов, если отказ временный.

    Возвращает:
        Одну фразу с рекомендацией.
    """
    if failure.is_temporary:
        return f"Сбой временный - {repeat_hint}."

    return "Сбой постоянный - повтор не поможет, ищи материал другим путём."


@tool(response_format = "content_and_artifact")
def search_web(query: str) -> tuple[str, str]:
    """Ищет страницы в интернете по запросу. Возвращает заголовок, адрес и краткое
    описание каждой найденной страницы."""
    outcome = search(query = query, max_results = _MAX_SEARCH_RESULTS)

    if outcome.failure:
        return (
            f"Поиск по запросу «{query}» не выполнен: {outcome.failure.reason}. "
            f"{_advice_on_failure(failure = outcome.failure, repeat_hint = 'повтори тот же запрос')}",
            CALL_FAILED,
        )

    if not outcome.results:
        return (
            f"По запросу «{query}» ничего не найдено. Попробуй другую формулировку.",
            CALL_COMPLETED,
        )

    listing = "\n\n".join(
        f"{position}. {item.title}\n{item.url}\n{item.snippet}"
        for position, item in enumerate(outcome.results, start = 1)
    )
    return (listing, CALL_COMPLETED)


@tool(response_format = "content_and_artifact")
def fetch_url(url: str) -> tuple[str, str]:
    """Скачивает страницу по адресу и возвращает её основной текст. Адрес брать
    только из выдачи search_web."""
    outcome = fetch_page(url = url, max_characters = _MAX_PAGE_CHARACTERS)

    if outcome.failure:
        return (
            f"Страница {url} не загрузилась: {outcome.failure.reason}. "
            f"{_advice_on_failure(failure = outcome.failure, repeat_hint = 'можно повторить тот же адрес')}",
            CALL_FAILED,
        )

    if not outcome.text:
        return (
            f"Из страницы {url} текст не получен: {outcome.empty_reason}. "
            "Возьми другой адрес из выдачи, этот повторять незачем.",
            CALL_COMPLETED,
        )

    return (f"Текст страницы {url}:\n\n{outcome.text}", CALL_COMPLETED)


RESEARCH_TOOLS = [search_web, fetch_url]
