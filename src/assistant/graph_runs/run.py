"""
Прогон графа: новый вопрос и продолжение записанного прогона.

Обновления состояния, приходящие от langgraph, разбираются здесь в шаги
ResearchStep: наружу уходит доменный шаг, а не сырой словарь обновления.

Состояние после каждого шага ложится в хранилище снимков под ключом run_id.
Тем же идентификатором называется файл журнала.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from assistant.graph import RESEARCH_TOOLS, Answer, ResearchNotes, ResearchState
from assistant.graph.budget import max_tool_calls_per_run
from assistant.graph_runs.checkpoints import open_checkpointer
from assistant.graph_runs.history import ResumePoint, find_resume_point
from assistant.graph_runs.logs import log_checkpoints_off, log_resume, log_run_id
from assistant.graph_runs.wiring import build_research_graph
from assistant.variables import CHECKPOINT_DIR

# Узлы, с которых можно продолжить прогон.
RESUMABLE_NODES = ("agent", "tools", "collect", "compose")


def new_run_id() -> str:
    """
    Заводит идентификатор прогона.

    Возвращает:
        Строку вида 20260829-160711.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_config(run_id: str, callbacks: list[BaseCallbackHandler]) -> dict:
    """
    Собирает конфиг вызова графа.

    Аргументы:
        run_id: ключ снимков; у продолжения тот же, что у исходного прогона.
        callbacks: слушатели прогона; журнал заводит вызывающий.

    Возвращает:
        Конфиг с ключом снимков, потолком рекурсии и слушателями.
    """
    # Запас по рекурсии: раунд стоит два шага (agent + tools), плюс финальный
    # agent и два узла вывода.
    max_calls = max_tool_calls_per_run(tools = RESEARCH_TOOLS)
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": max_calls * 2 + 5,
        "callbacks": callbacks,
    }


@dataclass(frozen = True)
class ResearchStep:
    """
    Шаг графа: чем закончился очередной узел.

    Атрибуты:
        node: имя узла.
        tool_calls: вызовы инструментов, объявленные узлом: пары «имя, аргументы».
        tool_results: исходы вызовов: пары «имя инструмента, исход вызова».
        notes: фактическая опора, если узел её дал; иначе None.
        answer: итоговый текст, если узел его дал; иначе None.
    """

    node: str
    tool_calls: list[tuple[str, dict]]
    tool_results: list[tuple[str, str]]
    notes: ResearchNotes | None
    answer: Answer | None


def _step_of(node: str, update: dict) -> ResearchStep:
    """
    Разбирает обновление состояния, пришедшее от узла.

    Аргументы:
        node: имя узла.
        update: обновление состояния, отданное узлом.

    Возвращает:
        Шаг графа с объявленными вызовами, их исходами, опорой и текстом.
    """
    messages = update.get("messages") or []

    tool_calls = [
        (call["name"], call["args"])
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]
    tool_results = [
        (message.name or "инструмент", str(message.artifact))
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    return ResearchStep(
        node = node,
        tool_calls = tool_calls,
        tool_results = tool_results,
        notes = update.get("notes"),
        answer = update.get("answer"),
    )


def _stream_steps(
    graph: CompiledStateGraph,
    initial_state: ResearchState | None,
    config: dict,
) -> Iterator[ResearchStep]:
    """
    Прогоняет граф и отдаёт шаг после каждого узла.

    Аргументы:
        graph: скомпилированный граф.
        initial_state: начальное состояние; None - продолжение с записанного снимка.
        config: конфиг вызова графа.

    Возвращает:
        Шаги графа по одному в порядке прохождения узлов.
    """
    for update in graph.stream(initial_state, config = config, stream_mode = "updates"):
        # Под ключом лежит обновление состояния от узла, но служебные ключи
        # langgraph несут не словарь, и разбирать их нечего.
        for node, state_update in update.items():
            if isinstance(state_update, dict):
                yield _step_of(node = node, update = state_update)


def run_research_staged(
    question: str,
    narrator_prompt: str | None,
    run_id: str,
    callbacks: list[BaseCallbackHandler],
) -> Iterator[ResearchStep]:
    """
    Прогоняет вопрос через граф, отдавая шаг после каждого узла.

    Аргументы:
        question: вопрос пользователя.
        narrator_prompt: блок про рассказчика для узла изложения; None -
            изложение без персонажа.
        run_id: идентификатор прогона: ключ снимков и имя файла журнала.
        callbacks: слушатели прогона.

    Возвращает:
        Шаги графа по одному в порядке прохождения узлов.
    """
    checkpointer = open_checkpointer(directory = CHECKPOINT_DIR)

    log_run_id(run_id = run_id)
    if checkpointer is None:
        log_checkpoints_off()

    initial_state: ResearchState = {
        "question": question,
        "messages": [HumanMessage(content = question)],
        "narrator_prompt": narrator_prompt,
        "notes": None,
        "answer": None,
    }

    yield from _stream_steps(
        graph = build_research_graph(checkpointer = checkpointer),
        initial_state = initial_state,
        config = _run_config(run_id = run_id, callbacks = callbacks),
    )


@dataclass(frozen = True)
class ResumedRun:
    """
    Исход продолжения записанного прогона.

    Атрибуты:
        answer: итоговый текст; None при неудаче.
        notes: фактическая опора; None при неудаче.
        question: вопрос пользователя из снимка; пустая строка при неудаче.
        narrator_prompt: блок про рассказчика из снимка; пустая строка, если
            рассказчик не задан, и при неудаче.
        error: причина неудачи; пустая строка при успехе.
    """

    answer: Answer | None
    notes: ResearchNotes | None
    question: str
    narrator_prompt: str
    error: str


def failed_resume(error: str) -> ResumedRun:
    """
    Собирает исход продолжения, оборвавшегося до вызова графа.

    Аргументы:
        error: причина неудачи.

    Возвращает:
        Исход без текста, опоры, вопроса и рассказчика.
    """
    return ResumedRun(
        answer = None,
        notes = None,
        question = "",
        narrator_prompt = "",
        error = error,
    )


def resume_research_staged(
    point: ResumePoint,
    narrator_prompt: str | None,
    callbacks: list[BaseCallbackHandler],
) -> Iterator[ResearchStep]:
    """
    Продолжает записанный прогон с найденной точки, отдавая шаг после каждого узла.

    Аргументы:
        point: точка входа, найденная find_resume_point.
        narrator_prompt: новый блок про рассказчика; None - взять из снимка.
        callbacks: слушатели прогона.

    Возвращает:
        Шаги графа по одному в порядке прохождения узлов.
    """
    graph = build_research_graph(checkpointer = open_checkpointer(directory = CHECKPOINT_DIR))

    log_resume(run_id = point.run_id, from_node = point.from_node)

    resume_from = point.config
    if narrator_prompt is not None:
        resume_from = graph.update_state(resume_from, values = {"narrator_prompt": narrator_prompt})

    # Слияние поузловое: в configurable снимка лежит checkpoint_id, без него
    # прогон пошёл бы с последнего состояния, а не с выбранного.
    config = _run_config(run_id = point.run_id, callbacks = callbacks)
    config["configurable"] = {**config["configurable"], **resume_from["configurable"]}

    yield from _stream_steps(graph = graph, initial_state = None, config = config)


def resume_research(
    run_id: str,
    from_node: str,
    narrator_prompt: str | None,
    callbacks: list[BaseCallbackHandler],
) -> ResumedRun:
    """
    Продолжает записанный прогон с указанного узла.

    Промежуточные шаги не нужны: возвращается только итог.

    Аргументы:
        run_id: идентификатор прогона.
        from_node: узел, с которого продолжать.
        narrator_prompt: новый блок про рассказчика; None - взять из снимка.
        callbacks: слушатели прогона.

    Возвращает:
        Исход продолжения: итоговый текст с опорой, вопрос и рассказчик из
        снимка либо причину неудачи.
    """
    point = find_resume_point(run_id = run_id, from_node = from_node)
    if point.error:
        return failed_resume(error = point.error)

    notes = point.notes
    answer = None

    for step in resume_research_staged(
        point = point,
        narrator_prompt = narrator_prompt,
        callbacks = callbacks,
    ):
        if step.notes is not None:
            notes = step.notes
        if step.answer is not None:
            answer = step.answer

    return ResumedRun(
        answer = answer,
        notes = notes,
        question = point.question,
        narrator_prompt = narrator_prompt or point.narrator_prompt,
        error = "",
    )
