"""
Контракт графа с внешним миром.

Клиенты моделей граф не создаёт: они приходят снаружи готовыми. Инструменты,
наоборот, его собственные - их имена стоят в промпте ресёрчера, а лимиты
подобраны под них; параметром они передаются только затем, чтобы тест мог
подставить свои.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from assistant.graph.state import ResearchState

# Узел графа: принимает состояние, возвращает его обновление.
ResearchNode = Callable[[ResearchState], dict]


@dataclass(frozen = True)
class NodeLlms:
    """
    Клиенты моделей по узлам графа.

    Схемы структурированного вывода навешивают сами узлы, поэтому клиенты
    приходят без них.

    Атрибуты:
        agent: клиент узла agent, выбирающего инструменты.
        collect: клиент узла collect, выжимающего факты.
        compose: клиент узла compose, излагающего текст.
    """

    agent: BaseChatModel
    collect: BaseChatModel
    compose: BaseChatModel
