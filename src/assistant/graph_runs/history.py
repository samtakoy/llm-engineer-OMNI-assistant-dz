"""
Чтение записанных прогонов: список переигрываемых и точка входа в снимок.

Модуль только смотрит в хранилище снимков и ничего не исполняет: продолжить
прогон с найденной точки умеет runs.
"""

from dataclasses import dataclass

from assistant.graph.graph import build_graph
from assistant.graph.state import ResearchNotes
from assistant.graph_runs.checkpoints import open_checkpointer
from assistant.variables import CHECKPOINT_DIR


@dataclass(frozen = True)
class ResumePoint:
    """
    Точка входа в записанный прогон.

    Атрибуты:
        run_id: идентификатор прогона.
        from_node: узел, с которого пойдёт продолжение.
        config: конфиг снимка перед этим узлом; пустой при неудаче.
        question: вопрос пользователя из снимка; пустая строка при неудаче.
        narrator_prompt: блок про рассказчика из снимка; пустая строка, если
            рассказчик не задан, и при неудаче.
        notes: фактическая опора из снимка; None, если её там ещё нет.
        error: причина неудачи; пустая строка при успехе.
    """

    run_id: str
    from_node: str
    config: dict
    question: str
    narrator_prompt: str
    notes: ResearchNotes | None
    error: str


def _failed_point(run_id: str, from_node: str, error: str) -> ResumePoint:
    """
    Собирает точку входа, которую не удалось найти.

    Аргументы:
        run_id: идентификатор прогона.
        from_node: узел, с которого продолжали.
        error: причина неудачи.

    Возвращает:
        Точку входа без конфига, вопроса, рассказчика и опоры.
    """
    return ResumePoint(
        run_id = run_id,
        from_node = from_node,
        config = {},
        question = "",
        narrator_prompt = "",
        notes = None,
        error = error,
    )


def find_resume_point(run_id: str, from_node: str) -> ResumePoint:
    """
    Ищет снимок, с которого записанный прогон продолжится указанным узлом.

    Аргументы:
        run_id: идентификатор прогона.
        from_node: узел, с которого продолжать.

    Возвращает:
        Точку входа с вопросом, рассказчиком и опорой из снимка либо причину
        неудачи.
    """
    checkpointer = open_checkpointer(directory = CHECKPOINT_DIR)
    if checkpointer is None:
        return _failed_point(
            run_id = run_id,
            from_node = from_node,
            error = "хранилище снимков выключено",
        )

    graph = build_graph(checkpointer = checkpointer)
    snapshots = list(graph.get_state_history({"configurable": {"thread_id": run_id}}))
    if not snapshots:
        return _failed_point(
            run_id = run_id,
            from_node = from_node,
            error = f"снимков прогона {run_id} нет",
        )

    target = next((snapshot for snapshot in snapshots if snapshot.next == (from_node,)), None)
    if target is None:
        available = ", ".join(sorted({node for snapshot in snapshots for node in snapshot.next}))
        return _failed_point(
            run_id = run_id,
            from_node = from_node,
            error = f"в прогоне {run_id} нет входа в узел {from_node}; есть: {available}",
        )

    return ResumePoint(
        run_id = run_id,
        from_node = from_node,
        config = target.config,
        question = str(target.values.get("question", "")),
        narrator_prompt = target.values.get("narrator_prompt") or "",
        notes = target.values.get("notes"),
        error = "",
    )


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


def latest_run_id() -> str:
    """
    Отдаёт идентификатор свежего записанного прогона.

    Возвращает:
        Идентификатор прогона; пустую строку, если записанных прогонов нет.
    """
    runs = list_runs()
    return runs[0][0] if runs else ""
