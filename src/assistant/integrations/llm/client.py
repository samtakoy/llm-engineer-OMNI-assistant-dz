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


def build_provider_config(provider: str) -> ProviderConfig:
    """
    Собирает настройки провайдера из переменных окружения.

    Аргументы:
        provider: одно из значений PROVIDER_NAMES.

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
            model = LOCAL_MODEL,
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
            model = OPENROUTER_MODEL,
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
            model = OPENAI_MODEL,
        )

    raise RuntimeError(f"Неизвестный провайдер {provider!r}. Ожидается одно из {PROVIDER_NAMES}")


# Бюджет размышления в отладочном режиме, см. docs/SO_with_reasoning.md.
DEBUG_REASONING_EFFORT = "low"


@lru_cache(maxsize = 8)
def build_llm(
    role: NodeRole,
    is_debug_reasoning_on: bool,
    provider: str = LLM_PROVIDER,
) -> ChatOpenAI:
    """
    Создаёт клиент модели для узла графа.

    Параметры собираются слоями: профиль модели, поверх него общее перекрытие
    роли, поверх - перекрытие роли под эту модель, поверх - переменные окружения.

    Аргументы:
        role: характер работы узла, по нему берётся перекрытие из реестра ролей.
        is_debug_reasoning_on: запросить текст размышления. Отладочный режим,
            он накладывает ограничения на схему: docs/SO_with_reasoning.md.
        provider: провайдер, по умолчанию из окружения.

    Возвращает:
        Готовый клиент.
    """
    config = build_provider_config(provider = provider)
    profile = profile_for(model = config.model).for_role(role = role)

    settings: dict[str, Any] = profile.standard()

    if is_debug_reasoning_on:
        settings["reasoning"] = {
            "effort": DEBUG_REASONING_EFFORT,
            "summary": "detailed",
        }
        # Если есть reasoning = {"effort": "low"} → уходит на Responses API (устройство langchain).
        # Оба параметра принадлежат /v1/chat/completions. На /v1/responses их нет
        # Будет TypeError: Responses.create() got an unexpected keyword argument 'reasoning_effort'
        settings.pop("reasoning_effort", None)
        settings.pop("presence_penalty", None)

    # Переменная окружения перебивает и профиль, и роль: она для замеров, когда
    # нужен один и тот же режим на всём прогоне.
    if LLM_TEMPERATURE:
        settings["temperature"] = float(LLM_TEMPERATURE)

    extra_body: dict[str, Any] = {**config.extra_body, **profile.extra()}
    if LLM_SEED:
        extra_body["seed"] = int(LLM_SEED)

    return ChatOpenAI(
        base_url = config.base_url,
        api_key = SecretStr(config.api_key),
        model = config.model,
        timeout = 600,
        max_retries = 1,
        extra_body = extra_body or None,
        **settings,
    )


# Поля клиента, которые уезжают в тело запроса. Список берём из профиля.
# Поле reasoning профилю не принадлежит - его добавляет отладочный режим.
_DESCRIBED_FIELDS = standard_field_names() + ("reasoning",)


def describe_llm(llm: ChatOpenAI) -> str:
    """
    Описывает параметры собранного клиента одной строкой.

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
    return body or "параметры сервера"


def reasoning_text(message: object) -> str:
    """
    Достаёт текст размышления из ответа модели.

    Стандартное поле block["reasoning"] у локального сервера пустое: lm studio
    отдаёт не сводку в формате openai, а сырой текст. Langchain его не теряет -
    складывает в extras.content, откуда мы и берём.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Текст размышления либо пустую строку.
    """
    blocks = getattr(message, "content_blocks", None) or []
    parts: list[str] = []

    for block in blocks:
        if block.get("type") != "reasoning":
            continue
        if block.get("reasoning"):
            parts.append(str(block["reasoning"]))
            continue
        for chunk in block.get("extras", {}).get("content", []):
            if chunk.get("text"):
                parts.append(str(chunk["text"]))

    return "\n".join(parts).strip()
