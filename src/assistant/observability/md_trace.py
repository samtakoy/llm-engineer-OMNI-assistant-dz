"""
Журнал прогона в markdown: цепочка запросов, ответов и вызовов инструментов.

Слушает события LangChain из слота config["callbacks"], узлы графа о журнале не
знают. Файл на прогон: глазами читается спойлерами, машиной режется по
маркерам <!--LOG ...-->, невидимым в рендере.

Кирпич без привязки к проекту: всё, что знает о конкретном графе, приходит
параметрами конструктора. Зависимости - langchain_core и стандартная
библиотека.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# Виды записей: по ним режут журнал.
KIND_GRAPH = "graph_start"
KIND_GRAPH_END = "graph_end"
KIND_NODE = "node_start"
KIND_REQUEST = "llm_request"
KIND_RESPONSE = "llm_response"
KIND_REASONING = "reasoning"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_NOTE = "note"


class MarkdownTrace(BaseCallbackHandler):
    """
    Пишет ход прогона в markdown-файл.

    Имя узла приходит в metadata только у событий-начал; у on_llm_end и
    on_tool_end его нет, поэтому карта run_id -> узел держится в поле, а для
    концовок имя берётся оттуда или у родителя.
    """

    def __init__(
        self,
        path: Path,
        header_rows: list[str],
        describe_request: Callable[[Any], str] | None,
        summarize_result: Callable[[Any], str] | None,
    ) -> None:
        """
        Аргументы:
            path: файл журнала; каталог создаётся при необходимости.
            header_rows: строки шапки файла, готовые к печати.
            describe_request: достаёт из стартового состояния текст запроса
                пользователя; None - запрос отдельно не показывать.
            summarize_result: сводит итоговое состояние в одну строку;
                None - итог без сводки.
        """
        path.parent.mkdir(parents = True, exist_ok = True)
        self._path = path
        self._describe_request = describe_request
        self._summarize_result = summarize_result
        self._seq = 0
        self._nodes: dict[str, str] = {}
        self._tools: dict[str, str] = {}
        self._clocks: dict[str, float] = {}
        self._visits: dict[str, int] = {}
        self._last_node = ""
        self._root = ""
        self._started = datetime.now()

        self._write_header(header_rows = header_rows)

    # --- события LangChain ------------------------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id = None,
                       metadata = None, **kwargs) -> None:
        """Начало графа или узла."""
        node = (metadata or {}).get("langgraph_node")
        if not node:
            sections: list[tuple[str | None, str, str]] = []
            request = self._describe_request(inputs) if self._describe_request else ""
            if request:
                sections.append((None, _as_quote(request), "raw"))
            sections.append(("стартовое состояние", _pretty(inputs), "json"))

            # Корень запоминается: у on_chain_end метаданных нет, и завершение
            # графа отличается от завершения внутренней цепочки только по
            # совпадению идентификатора.
            self._root = str(run_id)
            self._clocks[self._root] = time.monotonic()
            self._block(KIND_GRAPH, "граф", run_id, parent_run_id, "прогон начат",
                        sections = sections)
            return

        self._nodes[str(run_id)] = node

        # На один вход в узел приходит несколько chain_start - на сам узел и на
        # внутренние цепочки, у каждого свой run_id. Повторы отсекаются
        # сравнением с предыдущим именем.
        if node == self._last_node:
            return
        self._last_node = node

        self._visits[node] = self._visits.get(node, 0) + 1
        self._append(f"\n## {node} · вход {self._visits[node]}\n\n")
        self._block(KIND_NODE, node, run_id, parent_run_id, f"узел {node}")

    def on_chain_end(self, outputs, *, run_id, parent_run_id = None, **kwargs) -> None:
        """Конец графа: итоговое состояние симметрично стартовому."""
        if str(run_id) != self._root:
            return

        summary = "прогон закончен"
        if self._summarize_result:
            summary += f" - {self._summarize_result(outputs)}"

        self._append("\n## итог\n\n")
        self._block(KIND_GRAPH_END, "граф", run_id, parent_run_id, summary,
                    sections = [("итоговое состояние", _pretty(outputs), "json")],
                    elapsed = self._elapsed(run_id))

    def on_chain_error(self, error, *, run_id, parent_run_id = None, **kwargs) -> None:
        """Падение графа: без записи файл обрывается на полуслове."""
        if str(run_id) != self._root:
            return

        self._append("\n## итог\n\n")
        self._block(KIND_GRAPH_END, "граф", run_id, parent_run_id,
                    f"прогон упал: {type(error).__name__}",
                    sections = [(None, _as_quote(str(error)), "raw")],
                    elapsed = self._elapsed(run_id))

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id = None,
                            metadata = None, invocation_params = None, **kwargs) -> None:
        """Запрос к модели: последнее сообщение развёрнуто, вся переписка свёрнута."""
        node = (metadata or {}).get("langgraph_node", self._node_of(parent_run_id))
        self._nodes[str(run_id)] = node
        self._clocks[str(run_id)] = time.monotonic()
        turn = messages[0] if messages else []

        sections: list[tuple[str | None, str, str]] = []
        # Выдача инструмента пропускается: она напечатана блоком tool_result
        # строкой выше.
        if turn and not _is_tool_message(turn[-1]):
            sections.append((None, _render_message(turn[-1]), "raw"))
        sections.append((f"вся переписка целиком ({len(turn)} сообщений)",
                         _pretty([_describe(message) for message in turn]), "json"))

        if invocation_params:
            sections.append(("параметры вызова", _pretty(_slim_params(invocation_params)), "json"))

        self._block(KIND_REQUEST, node, run_id, parent_run_id,
                    f"запрос к модели ({len(turn)} сообщений)", sections = sections)

    def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs) -> None:
        """Ответ модели: текст, размышление и запрошенные инструменты."""
        node = self._node_of(run_id) or self._node_of(parent_run_id)
        message = _first_message(response)
        if message is None:
            return

        elapsed = self._elapsed(run_id)

        reasoning = _reasoning_text(message)
        if reasoning:
            self._block(KIND_REASONING, node, run_id, parent_run_id, "размышление",
                        sections = [(None, _as_quote(reasoning), "raw")])

        text = _answer_text(message)
        calls = getattr(message, "tool_calls", None) or []
        summary = "ответ модели"
        if calls:
            summary += ": " + ", ".join(call["name"] for call in calls)

        sections: list[tuple[str | None, str, str]] = []
        if text:
            # Под схемой модель отвечает json, а не прозой: раскладывается,
            # чтобы поля читались.
            parsed = _as_json(text)
            sections.append((None, f"```json\n{parsed}\n```", "raw") if parsed
                            else (None, _as_quote(text), "raw"))
        for call in calls:
            sections.append((None, f"`{call['name']}({_call_args(call)})`", "raw"))
        sections.append(("сырой ответ", _pretty({"text": text, "tool_calls": calls}), "json"))

        self._block(KIND_RESPONSE, node, run_id, parent_run_id, summary,
                    sections = sections, elapsed = elapsed)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id = None,
                      metadata = None, inputs = None, **kwargs) -> None:
        """Вызов инструмента."""
        name = (serialized or {}).get("name", "?")
        node = (metadata or {}).get("langgraph_node", self._node_of(parent_run_id))
        self._nodes[str(run_id)] = node
        self._tools[str(run_id)] = name
        self._block(KIND_TOOL_CALL, node, run_id, parent_run_id, f"вызов {name}",
                    body = _pretty(inputs if inputs is not None else input_str))

    def on_tool_end(self, output, *, run_id, parent_run_id = None, **kwargs) -> None:
        """Выдача инструмента целиком, без обрезки."""
        name = self._tools.get(str(run_id), "инструмент")
        node = self._node_of(run_id) or self._node_of(parent_run_id)
        self._block(KIND_TOOL_RESULT, node, run_id, parent_run_id, f"выдача {name}",
                    body = _pretty(getattr(output, "content", output)))

    # --- служебное --------------------------------------------------------

    def note(self, text: str) -> None:
        """
        Пишет запись приложения - то, что не является событием LangChain.

        Аргументы:
            text: текст записи. Относится к последнему начатому узлу.
                Первая строка идёт в спину журнала, остальные - под неё.

        Возвращает:
            Ничего.
        """
        head, _, tail = text.strip().partition("\n")
        sections = [(None, _as_quote(tail), "raw")] if tail.strip() else None
        self._block(KIND_NOTE, self._last_node or "граф", None, None, head,
                    sections = sections)

    def _node_of(self, run_id: UUID | None) -> str:
        """
        Возвращает имя узла по идентификатору вызова.

        Аргументы:
            run_id: идентификатор вызова либо None.

        Возвращает:
            Имя узла либо прочерк.
        """
        return self._nodes.get(str(run_id), "—")

    def _write_header(self, header_rows: list[str]) -> None:
        """
        Пишет шапку файла: когда прогон и на чём.

        Аргументы:
            header_rows: строки шапки, готовые к печати.

        Возвращает:
            Ничего.
        """
        stamp = self._started.isoformat(timespec = "seconds")
        rows = [f"# Прогон {stamp}", "", *header_rows, ""]
        self._append("\n".join(rows) + "\n")

    def _block(self, kind: str, node: str, run_id: UUID | None, parent_run_id: UUID | None,
               summary: str, body: str = "", sections: list[tuple[str | None, str, str]] | None = None,
               elapsed: float | None = None) -> None:
        """
        Пишет одну запись журнала.

        Аргументы:
            kind: вид записи, по нему фильтруют.
            node: имя узла.
            run_id: идентификатор вызова.
            parent_run_id: идентификатор родителя, по нему собирается дерево.
            summary: одна строка для чтения глазами.
            body: тело одним куском; попадёт под спойлер как json.
            sections: части тела «заголовок, текст, стиль». Заголовок None -
                печатается сразу, без спойлера; стиль json заворачивает в блок
                кода, raw оставляет как есть.
            elapsed: длительность вызова в секундах, если применимо.

        Возвращает:
            Ничего.
        """
        self._seq += 1
        clock = datetime.now().strftime("%H:%M:%S")
        marker = (
            f"<!--LOG seq={self._seq} ts={datetime.now().isoformat(timespec = 'seconds')} "
            f"kind={kind} node={node} run={run_id or '—'} parent={parent_run_id or '—'}-->"
        )

        spine = f"- `{clock}` **{node}** → {summary}"
        if elapsed is not None:
            spine += f" _({elapsed:.1f} с)_"
        lines = [spine, marker]

        parts = list(sections or [])
        if body:
            parts.append((summary, body, "json"))

        for title, text, style in parts:
            rendered = f"```json\n{text}\n```" if style == "json" else text
            if title is None:
                lines.append("")
                lines.append(rendered)
            else:
                lines.append(f"<details><summary>{title}</summary>\n")
                lines.append(rendered)
                lines.append("\n</details>")

        lines.append("<!--/LOG-->\n")
        self._append("\n".join(lines) + "\n")

    def _elapsed(self, run_id: UUID) -> float | None:
        """
        Возвращает длительность вызова по его идентификатору.

        Аргументы:
            run_id: идентификатор вызова.

        Возвращает:
            Секунды либо None, если начало не зафиксировано.
        """
        started = self._clocks.pop(str(run_id), None)
        return None if started is None else time.monotonic() - started

    def _append(self, text: str) -> None:
        """
        Дописывает текст в файл, закрывая его сразу.

        Аргументы:
            text: что дописать.

        Возвращает:
            Ничего.
        """
        with self._path.open("a", encoding = "utf-8") as handle:
            handle.write(text)


class NoteHandler(logging.Handler):
    """Уводит записи logging в тот же журнал."""

    def __init__(self, trace: MarkdownTrace) -> None:
        """
        Аргументы:
            trace: журнал, куда писать.
        """
        super().__init__(level = logging.INFO)
        self._trace = trace

    def emit(self, record: logging.LogRecord) -> None:
        """
        Аргументы:
            record: запись logging.

        Возвращает:
            Ничего.
        """
        self._trace.note(text = self.format(record))


def _as_quote(text: str) -> str:
    """
    Оформляет прозу цитатой.

    Аргументы:
        text: текст модели.

    Возвращает:
        Текст, где каждая строка начинается с «> ».
    """
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _as_json(text: str) -> str | None:
    """
    Разбирает текст как json, если он им является.

    Аргументы:
        text: текст ответа модели.

    Возвращает:
        Разложенный json либо None, если это проза.
    """
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.dumps(json.loads(stripped), ensure_ascii = False, indent = 2)
    except (TypeError, ValueError):
        return None


def _call_args(call: dict[str, Any]) -> str:
    """
    Собирает аргументы вызова в одну читаемую строку.

    Аргументы:
        call: вызов инструмента.

    Возвращает:
        Строку вида «query=гора машук».
    """
    args = call.get("args") or {}
    if not isinstance(args, dict):
        return str(args)
    return ", ".join(f"{key}={value}" for key, value in args.items() if value is not None)


def _slim_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Оставляет из параметров вызова то, что проверяют по журналу.

    Аргументы:
        params: invocation_params из колбэка.

    Возвращает:
        Урезанный словарь.
    """
    keys = ("model", "temperature", "top_p", "presence_penalty", "reasoning_effort",
            "response_format", "tool_choice", "stop", "extra_body")
    slim = {key: params[key] for key in keys if key in params}
    tools = params.get("tools")
    if tools:
        slim["tools"] = [tool.get("function", {}).get("name", "?") for tool in tools]
    return slim


def _is_tool_message(message: object) -> bool:
    """
    Отличает выдачу инструмента от остальных сообщений.

    Аргументы:
        message: сообщение LangChain.

    Возвращает:
        True для ToolMessage.
    """
    return type(message).__name__ == "ToolMessage"


def _render_message(message: object) -> str:
    """
    Печатает одно сообщение диалога читаемо.

    Аргументы:
        message: сообщение LangChain.

    Возвращает:
        Markdown с ролью и телом.
    """
    described = _describe(message)
    role = described["role"]
    text = described.get("text") or ""

    lines = [f"**последнее сообщение - {role}**", ""]
    if text:
        parsed = _as_json(text)
        lines.append(f"```json\n{parsed}\n```" if parsed else _as_quote(text))
    calls = described.get("tool_calls")
    if calls:
        lines.append("")
        lines.extend(f"`{call['name']}({_call_args(call)})`" for call in calls)
    return "\n".join(lines)


def _describe(message: object) -> dict[str, Any]:
    """
    Сжимает сообщение до того, что важно в журнале.

    Аргументы:
        message: сообщение LangChain.

    Возвращает:
        Словарь с ролью, текстом и вызовами инструментов.
    """
    row: dict[str, Any] = {
        "role": type(message).__name__.replace("Message", "").lower(),
        "text": _answer_text(message),
    }
    calls = getattr(message, "tool_calls", None)
    if calls:
        row["tool_calls"] = calls
    return row


def _first_message(response: object) -> object | None:
    """
    Достаёт сообщение модели из результата вызова.

    Аргументы:
        response: LLMResult.

    Возвращает:
        Сообщение либо None.
    """
    generations = getattr(response, "generations", None) or []
    for row in generations:
        for generation in row:
            message = getattr(generation, "message", None)
            if message is not None:
                return message
    return None


def _answer_text(message: object) -> str:
    """
    Достаёт текст ответа модели, минуя блоки размышления.

    Аргументы:
        message: ответ модели.

    Возвращает:
        Текст ответа либо пустую строку.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()

    parts = [
        str(block["text"])
        for block in (getattr(message, "content_blocks", None) or [])
        if block.get("type") == "text" and block.get("text")
    ]
    return "\n".join(parts).strip()


def _reasoning_text(message: object) -> str:
    """
    Достаёт текст размышления из ответа модели.

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


def _pretty(value: object) -> str:
    """
    Печатает значение читаемо, не падая на том, что не сериализуется.

    Аргументы:
        value: что печатать.

    Возвращает:
        Строку.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return value

    try:
        return json.dumps(value, ensure_ascii = False, indent = 2, default = _fallback)
    except (TypeError, ValueError):
        return str(value)


def _fallback(value: object) -> object:
    """
    Разворачивает то, что json не умеет сам.

    Аргументы:
        value: несериализуемое значение.

    Возвращает:
        Словарь для модели pydantic, иначе строку.
    """
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else str(value)
