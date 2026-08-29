"""
Клиент модели для графа.

Провайдер выбирается переменной окружения, весь код ходит через один
openai-совместимый интерфейс. Переезд на облако или на vllm - смена двух
переменных, а не рефакторинг.

Параметры сэмплирования берутся из соседнего модуля profiles и уезжают в теле
запроса, перекрывая настройки сервера. Так забыть выставить их в ui lm studio
становится нечем.

Узел графа задаёт не параметры, а свою роль: чем занят узел, знает он сам, а
какими настройками это делается - знает реестр ролей. Иначе список аргументов
клиента растёт с каждым параметром, который понадобилось перекрыть одному узлу.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from assistant.integrations.llm.profiles import (
    NodeRole,
    profile_for,
    standard_field_names,
)
from assistant.variables import (
    LLM_PROVIDER,
    LLM_SEED,
    LLM_TEMPERATURE,
    LOCAL_API_KEY,
    LOCAL_BASE_URL,
    LOCAL_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

PROVIDER_NAMES = ("local", "openrouter", "openai")


@dataclass(frozen = True)
class ProviderConfig:
    """
    Настройки провайдера моделей.

    Атрибуты:
        name: имя провайдера.
        base_url: адрес openai-совместимого сервера.
        api_key: ключ доступа.
        model: имя модели как его знает сервер.
        extra_body: дополнительные поля запроса, специфичные для провайдера.
    """

    name: str
    base_url: str
    api_key: str
    model: str
    extra_body: dict[str, Any] = field(default_factory = dict)


def _require(value: str, variable: str, provider: str) -> str:
    """
    Проверяет, что обязательная переменная заполнена.

    Аргументы:
        value: значение переменной.
        variable: имя переменной для текста ошибки.
        provider: провайдер, которому она нужна.

    Возвращает:
        Непустое значение.

    Исключения:
        RuntimeError: если переменная пуста.
    """
    if not value:
        raise RuntimeError(f"Не задана {variable} (провайдер {provider}). Заполни .env")
    return value


def build_provider_config(provider: str, model: str | None) -> ProviderConfig:
    """
    Собирает настройки провайдера из переменных окружения.

    Аргументы:
        provider: одно из значений PROVIDER_NAMES.
        model: имя модели, которое перекроет взятое из окружения. None -
            оставить модель провайдера по умолчанию.

    Возвращает:
        Конфигурацию провайдера.

    Исключения:
        RuntimeError: если провайдер неизвестен или его переменные пусты.
    """
    if provider == "local":
        # У локального сервера ключ формальный, непустым его не требуем.
        return ProviderConfig(
            name = "local",
            base_url = LOCAL_BASE_URL,
            api_key = LOCAL_API_KEY or "not-needed",
            model = model or LOCAL_MODEL,
        )

    if provider == "openrouter":
        return ProviderConfig(
            name = "openrouter",
            base_url = OPENROUTER_BASE_URL,
            api_key = _require(
                value = OPENROUTER_API_KEY,
                variable = "OPENROUTER_API_KEY",
                provider = provider,
            ),
            model = model or OPENROUTER_MODEL,
            extra_body = {"reasoning": {"enabled": False}},
        )

    if provider == "openai":
        return ProviderConfig(
            name = "openai",
            base_url = OPENAI_BASE_URL,
            api_key = _require(
                value = OPENAI_API_KEY,
                variable = "OPENAI_API_KEY",
                provider = provider,
            ),
            model = model or OPENAI_MODEL,
        )

    raise RuntimeError(f"Неизвестный провайдер {provider!r}. Ожидается одно из {PROVIDER_NAMES}")


# Бюджет размышления, когда отладочный режим включает его на всех узлах.
FORCED_REASONING_EFFORT = "low"

# Поле ответа с текстом размышления. В спецификацию openai не входит: его шлют
# openai-совместимые серверы, langchain такие поля отбрасывает.
REASONING_FIELD = "reasoning_content"


class ChatOpenAIWithReasoning(ChatOpenAI):
    """
    Клиент чата, сохраняющий текст размышления.

    Базовый класс отбрасывает поля ответа вне спецификации openai и отсылает к
    подклассу провайдера. Здесь reasoning_content переносится в
    additional_kwargs сообщения.
    """

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """
        Собирает результат вызова и добавляет к сообщениям текст размышления.

        Аргументы:
            response: ответ сервера.
            generation_info: сведения о генерации.

        Возвращает:
            Результат вызова.
        """
        result = super()._create_chat_result(
            response = response,
            generation_info = generation_info,
        )

        for generation, choice in zip(result.generations, _choices(response = response)):
            reasoning = _choice_reasoning(choice = choice)
            if reasoning:
                generation.message.additional_kwargs[REASONING_FIELD] = reasoning

        return result


def _choices(response: Any) -> list[Any]:
    """
    Достаёт варианты ответа сервера.

    Аргументы:
        response: ответ сервера словарём либо объектом клиента openai.

    Возвращает:
        Список вариантов ответа.
    """
    if isinstance(response, dict):
        return list(response.get("choices") or [])

    return list(getattr(response, "choices", None) or [])


def _choice_reasoning(choice: Any) -> str:
    """
    Достаёт текст размышления из одного варианта ответа.

    Аргументы:
        choice: вариант ответа словарём либо объектом клиента openai.

    Возвращает:
        Текст размышления либо пустую строку.
    """
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if message is None:
        return ""

    value = (message.get(REASONING_FIELD) if isinstance(message, dict)
             else getattr(message, REASONING_FIELD, None))

    return str(value).strip() if value else ""


@lru_cache(maxsize = 8)
def build_llm(
    role: NodeRole,
    is_reasoning_forced: bool,
    model: str | None,
    provider: str = LLM_PROVIDER,
) -> ChatOpenAI:
    """
    Создаёт клиент модели для узла графа.

    Параметры собираются слоями: профиль модели, поверх него общее перекрытие
    роли, поверх - перекрытие роли под эту модель, поверх - переменные окружения.

    Аргументы:
        role: характер работы узла, по нему берётся перекрытие из реестра ролей.
        is_reasoning_forced: включить размышление независимо от профиля модели
            и роли узла. Отладочный режим для разбора поведения модели.
        model: имя модели. None - взять модель провайдера из окружения.
        provider: провайдер, по умолчанию из окружения.

    Возвращает:
        Готовый клиент.
    """
    config = build_provider_config(provider = provider, model = model)
    profile = profile_for(model = config.model).for_role(role = role)

    settings: dict[str, Any] = profile.standard()

    if is_reasoning_forced:
        settings["reasoning_effort"] = FORCED_REASONING_EFFORT

    # Переменная окружения перебивает и профиль, и роль: она для замеров, когда
    # нужен один и тот же режим на всём прогоне.
    if LLM_TEMPERATURE:
        settings["temperature"] = float(LLM_TEMPERATURE)

    extra_body: dict[str, Any] = {**config.extra_body, **profile.extra()}
    if LLM_SEED:
        extra_body["seed"] = int(LLM_SEED)

    return ChatOpenAIWithReasoning(
        base_url = config.base_url,
        api_key = SecretStr(config.api_key),
        model = config.model,
        timeout = 600,
        max_retries = 1,
        extra_body = extra_body or None,
        **settings,
    )


# Поля клиента, которые уезжают в тело запроса. Список берём из профиля.
_DESCRIBED_FIELDS = standard_field_names()


def describe_llm(llm: ChatOpenAI) -> str:
    """
    Описывает модель и параметры собранного клиента одной строкой.

    Читает поля самого клиента, а не профиль модели: роль узла перекрывает
    профиль, и строка должна показывать то, что реально уедет на сервер.

    Аргументы:
        llm: клиент, собранный build_llm.

    Возвращает:
        Строку для вывода в командной строке.
    """
    settings: dict[str, Any] = {}

    for name in _DESCRIBED_FIELDS:
        value = getattr(llm, name, None)
        if value is not None:
            settings[name] = value

    settings.update(llm.model_kwargs or {})
    settings.update(llm.extra_body or {})

    body = ", ".join(f"{key}={value}" for key, value in settings.items())
    return f"{llm.model_name} | {body or 'параметры сервера'}"


def reasoning_text(message: object) -> str:
    """
    Достаёт текст размышления из ответа модели.

    Текст лежит в additional_kwargs: туда его кладёт ChatOpenAIWithReasoning.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Текст размышления либо пустую строку.
    """
    extras = getattr(message, "additional_kwargs", None) or {}

    return str(extras.get(REASONING_FIELD, "")).strip()
