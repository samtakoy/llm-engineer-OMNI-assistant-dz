"""
Граф ресёрчера.

Цикл react на сборе фактов, затем два отдельных узла на выходе:

    START -> agent -(есть вызовы инструментов)-> tools -> agent
               \\-(вызовов нет)-> collect -> compose -> END

Узел agent ищет, collect выжимает из найденного проверяемые факты, compose
излагает их в запрошенном пользователем виде.

Состояние после каждого шага ложится в хранилище снимков под ключом run_id.
Тем же идентификатором называется файл журнала.
"""

from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from assistant.graph.budget import MAX_TOOL_CALLS_PER_RUN
from assistant.graph.checkpoints import open_checkpointer
from assistant.graph.llms import describe_nodes
from assistant.graph.logs import log_checkpoints_off, log_resume, log_run_id
from assistant.graph.nodes import agent_node, collect_node, compose_node, tools_node
from assistant.graph.state import Answer, ResearchNotes, ResearchState
from assistant.observability.tracing import build_callbacks
from assistant.variables import CHECKPOINT_DIR

# Узлы, с которых можно продолжить прогон.
RESUMABLE_NODES = ("agent", "tools", "collect", "compose")


def _route_after_agent(state: ResearchState) -> str:
    """
    Выбирает следующий узел по последнему сообщению модели.

    Аргументы:
        state: текущее состояние графа.

    Возвращает:
        Имя следующего узла: tools или collect.
    """
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else "collect"


def build_graph(checkpointer: BaseCheckpointSaver | None):
    """
    Собирает и компилирует граф ресёрчера.

    Аргументы:
        checkpointer: хранилище снимков состояния; None - без снимков.

    Возвращает:
        Скомпилированный граф, готовый к invoke.
    """
    builder = StateGraph(ResearchState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("collect", collect_node)
    builder.add_node("compose", compose_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent, ["tools", "collect"])
    builder.add_edge("tools", "agent")
    builder.add_edge("collect", "compose")
    builder.add_edge("compose", END)

    return builder.compile(checkpointer = checkpointer)


def _new_run_id() -> str:
    """
    Заводит идентификатор прогона.

    Возвращает:
        Строку вида 20260829-160711.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_config(run_id: str, trace_id: str, origin_rows: list[str]) -> dict:
    """
    Собирает конфиг вызова графа.

    Аргументы:
        run_id: ключ снимков; у продолжения тот же, что у исходного прогона.
        trace_id: имя файла журнала; у продолжения своё.
        origin_rows: строки о происхождении прогона для шапки журнала.

    Возвращает:
        Конфиг с ключом снимков, потолком рекурсии и слушателями журнала.
    """
    # Запас по рекурсии: раунд стоит два шага (agent + tools), плюс финальный
    # agent и два узла вывода.
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": MAX_TOOL_CALLS_PER_RUN * 2 + 5,
        "callbacks": build_callbacks(
            node_rows = describe_nodes(),
            trace_id = trace_id,
            origin_rows = origin_rows,
        ),
    }


def run_research(question: str, narrator_prompt: str | None) -> tuple[Answer, ResearchNotes]:
    """
    Прогоняет вопрос через граф.

    Аргументы:
        question: вопрос пользователя.
        narrator_prompt: блок про рассказчика для узла изложения; None -
            изложение без персонажа.

    Возвращает:
        Кортеж из итогового текста и фактической опоры, на которой он построен.
    """
    run_id = _new_run_id()
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

    final_state = build_graph(checkpointer = checkpointer).invoke(
        initial_state,
        config = _run_config(run_id = run_id, trace_id = run_id, origin_rows = []),
    )

    return final_state["answer"], final_state["notes"]


def resume_research(
    run_id: str,
    from_node: str,
    narrator_prompt: str | None,
) -> tuple[Answer | None, ResearchNotes | None, str]:
    """
    Продолжает записанный прогон с указанного узла.

    Аргументы:
        run_id: идентификатор прогона.
        from_node: узел, с которого продолжать.
        narrator_prompt: новый блок про рассказчика; None - взять из снимка.

    Возвращает:
        Кортеж из итогового текста, фактической опоры и причины неудачи.
        При неудаче первые два значения - None.
    """
    checkpointer = open_checkpointer(directory = CHECKPOINT_DIR)
    if checkpointer is None:
        return None, None, "хранилище снимков выключено"

    graph = build_graph(checkpointer = checkpointer)
    snapshots = list(graph.get_state_history({"configurable": {"thread_id": run_id}}))
    if not snapshots:
        return None, None, f"снимков прогона {run_id} нет"

    target = next((snapshot for snapshot in snapshots if snapshot.next == (from_node,)), None)
    if target is None:
        available = ", ".join(sorted({node for snapshot in snapshots for node in snapshot.next}))
        return None, None, f"в прогоне {run_id} нет входа в узел {from_node}; есть: {available}"

    log_resume(run_id = run_id, from_node = from_node)

    resume_from = target.config
    if narrator_prompt is not None:
        resume_from = graph.update_state(resume_from, values = {"narrator_prompt": narrator_prompt})

    # Журнал у продолжения свой: имя исходного прогона плюс время рестарта.
    # Исходный файл остаётся нетронутым, а происхождение видно и в имени, и в шапке.
    trace_id = f"{run_id}+{_new_run_id()}"
    origin_rows = [f"- продолжение прогона `{run_id}` с узла `{from_node}`"]

    # Слияние поузловое: в configurable снимка лежит checkpoint_id, без него
    # прогон пошёл бы с последнего состояния, а не с выбранного.
    config = _run_config(run_id = run_id, trace_id = trace_id, origin_rows = origin_rows)
    config["configurable"] = {**config["configurable"], **resume_from["configurable"]}

    final_state = graph.invoke(None, config = config)

    return final_state["answer"], final_state["notes"], ""


def list_runs() -> list[tuple[str, str]]:
    """
    Перечисляет прогоны, которые можно переиграть.

    Возвращает:
        Список пар «идентификатор прогона, вопрос пользователя», свежие первыми.
        Пустой список, если снимков нет.
    """
    checkpointer = open_checkpointer(directory = CHECKPOINT_DIR)
    if checkpointer is None:
        return []

    questions: dict[str, str] = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id in questions:
            continue
        question = checkpoint.checkpoint.get("channel_values", {}).get("question", "")
        questions[thread_id] = str(question)

    return sorted(questions.items(), reverse = True)
