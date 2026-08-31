"""
Тесты сборки настроек провайдера. Сеть не трогается, модели не запускаются.
"""

import pytest

from assistant.integrations.llm.client import (
    PROVIDER_NAMES,
    _yandex_model_uri,
    build_provider_config,
)
from assistant.variables import LOCAL_BASE_URL, OLLAMA_BASE_URL, OLLAMA_MODEL


def test_ollama_reads_own_variables() -> None:
    """
    Проверяет, что провайдер ollama берёт адрес и модель из своих переменных.
    """
    config = build_provider_config(provider = "ollama", model = None)

    assert config.base_url == OLLAMA_BASE_URL
    assert config.model == OLLAMA_MODEL


def test_reasoning_field_belongs_to_provider() -> None:
    """
    Проверяет имена поля с размышлением: у lm studio и у ollama они разные.
    """
    assert build_provider_config(provider = "local", model = None).reasoning_field == "reasoning_content"
    assert build_provider_config(provider = "ollama", model = None).reasoning_field == "reasoning"


def test_local_keeps_own_address() -> None:
    """
    Проверяет, что провайдер ollama не перетянул на себя адрес lm studio.
    """
    assert build_provider_config(provider = "local", model = None).base_url == LOCAL_BASE_URL
    assert OLLAMA_BASE_URL != LOCAL_BASE_URL


def test_every_named_provider_builds() -> None:
    """
    Проверяет, что каждое имя из PROVIDER_NAMES собирается либо внятно ругается
    на незаполненный ключ, но не падает неизвестным провайдером.
    """
    for name in PROVIDER_NAMES:
        try:
            config = build_provider_config(provider = name, model = None)
        except RuntimeError as error:
            assert "Не задана" in str(error)
            continue

        assert config.name == name
        assert config.reasoning_field


def test_yandex_model_becomes_uri() -> None:
    """
    Проверяет, что имя модели yandex cloud превращается в uri с каталогом, а
    готовый uri остаётся как есть.
    """
    assert _yandex_model_uri(model = "yandexgpt-lite", folder_id = "b1g0") == "gpt://b1g0/yandexgpt-lite"
    assert _yandex_model_uri(model = "gpt://иной/имя", folder_id = "b1g0") == "gpt://иной/имя"


def test_unknown_provider_is_rejected() -> None:
    """
    Проверяет, что неизвестное имя провайдера отвергается.
    """
    with pytest.raises(RuntimeError):
        build_provider_config(provider = "нет-такого", model = None)
