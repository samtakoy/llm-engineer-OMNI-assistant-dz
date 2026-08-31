"""
Тесты рестарта прогона со снимка. Модели не запускаются, инструмент поддельный.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from assistant.graph import CALL_COMPLETED, NodeLlms, build_graph
from assistant.graph.state import Answer, ResearchNotes, ResearchState, Section


@tool(response_format = "content_and_artifact")
def search_web(query: str) -> tuple[str, str]:
    """Ищет страницы в интернете по запросу. Возвращает заготовленную выдачу."""
    return "выдача", CALL_COMPLETED


class FakeStructured:
    """Клиент со схемой, отвечающий заготовкой."""

    def __init__(self, schema) -> None:
        self._schema = schema

    def invoke(self, messages) -> ResearchNotes | Answer:
        """
        Аргументы:
            messages: переписка, которая ушла бы в модель.

        Возвращает:
            Заготовку по схеме узла.
        """
        if self._schema is ResearchNotes:
            return ResearchNotes(
                summary = "сводка",
                facts = ["первый факт", "второй факт"],
                details = ["первая подробность"],
                gaps = ["первый пропуск"],
                handoff = "заметка ресёрчера",
                confidence = "высокая",
                sources = ["https://example.org"],
            )

        return Answer(
            title = "заголовок",
            intro = "вступление",
            sections = [Section(title = "раздел", content = "содержание")],
            closing = "завершение",
        )


class FakeLLM:
    """Клиент модели, отвечающий по заранее заданному списку."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies

    def bind_tools(self, tools):
        """
        Аргументы:
            tools: инструменты, которые привязал бы узел.

        Возвращает:
            Себя же.
        """
        return self

    def with_structured_output(self, schema, method) -> FakeStructured:
        """
        Аргументы:
            schema: схема ответа.
            method: способ, которым запрашивают схему.

        Возвращает:
            Клиент со схемой.
        """
        return FakeStructured(schema = schema)

    def invoke(self, messages) -> AIMessage:
        """
        Аргументы:
            messages: переписка, которая ушла бы в модель.

        Возвращает:
            Очередной ответ из списка.
        """
        if not self._replies:
            raise AssertionError("узел agent вызван, хотя прогон начат после него")

        return self._replies.pop(0)


def build_start_state() -> ResearchState:
    """
    Собирает стартовое состояние для тестов.

    Возвращает:
        Состояние с одним вопросом пользователя.
    """
    return {
        "question": "вопрос",
        "messages": [HumanMessage(content = "вопрос")],
        "narrator_prompt": "исходный рассказчик",
        "notes": None,
        "answer": None,
    }


def build_fake_graph(saver: InMemorySaver, agent_replies: list[AIMessage]):
    """
    Собирает граф на поддельных клиентах и поддельном инструменте.

    Аргументы:
        saver: хранилище снимков.
        agent_replies: ответы, которые узел agent отдаст по порядку.

    Возвращает:
        Скомпилированный граф.
    """
    return build_graph(
        checkpointer = saver,
        llms = NodeLlms(
            agent = FakeLLM(replies = agent_replies),
            collect = FakeLLM(replies = []),
            compose = FakeLLM(replies = []),
        ),
        tools = [search_web],
    )


def run_once(saver: InMemorySaver, config: dict) -> dict:
    """
    Прогоняет граф целиком на поддельных клиентах.

    Аргументы:
        saver: хранилище снимков.
        config: конфиг вызова с ключом прогона.

    Возвращает:
        Итоговое состояние.
    """
    replies = [
        AIMessage(
            content = "",
            tool_calls = [{"name": "search_web", "args": {"query": "запрос"}, "id": "c1"}],
        ),
        AIMessage(content = "материала достаточно"),
    ]

    graph = build_fake_graph(saver = saver, agent_replies = replies)
    return graph.invoke(build_start_state(), config = config)


def test_history_holds_entry_into_every_node() -> None:
    """
    Проверяет, что в истории снимков есть вход в каждый узел графа.
    """
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "прогон"}}
    run_once(saver = saver, config = config)

    graph = build_fake_graph(saver = saver, agent_replies = [])
    entries = {node for snapshot in graph.get_state_history(config) for node in snapshot.next}

    assert {"agent", "tools", "collect", "compose"} <= entries


def test_resume_skips_agent() -> None:
    """
    Проверяет, что рестарт с compose не заходит в узел agent.
    """
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "прогон"}}
    run_once(saver = saver, config = config)

    # Пустой список ответов: заход в agent уронил бы тест исключением.
    graph = build_fake_graph(saver = saver, agent_replies = [])
    target = next(snapshot for snapshot in graph.get_state_history(config)
                  if snapshot.next == ("compose",))

    final_state = graph.invoke(None, config = target.config)

    assert final_state["answer"].title == "заголовок"
    assert len(final_state["notes"].facts) == 2


def test_resume_replaces_narrator() -> None:
    """
    Проверяет, что подмена рассказчика доезжает до узла изложения.
    """
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "прогон"}}
    run_once(saver = saver, config = config)

    # Пустой список ответов: заход в agent уронил бы тест исключением.
    graph = build_fake_graph(saver = saver, agent_replies = [])
    target = next(snapshot for snapshot in graph.get_state_history(config)
                  if snapshot.next == ("compose",))

    resume_from = graph.update_state(target.config, values = {"narrator_prompt": "другой рассказчик"})

    graph.invoke(None, config = resume_from)

    assert graph.get_state(config).values["narrator_prompt"] == "другой рассказчик"
