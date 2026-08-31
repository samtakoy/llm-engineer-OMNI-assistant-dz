"""
Показ исхода прогона: текст лекции, персонаж, опора и длительности в markdown.

Модуль не знает про gradio: наружу уходят готовые строки.
"""

from ..graph import Answer, ResearchNotes
from ..graph_runs import ResearchStep
from ..persona import Persona
from ..timing import Stopwatch


def render_answer(answer: Answer | None) -> str:
    """
    Собирает текст лекции.

    Аргументы:
        answer: итоговый текст; None - текста ещё нет.

    Возвращает:
        Заголовок, вступление, разделы и завершение в markdown. Пустую строку,
        если текста нет.
    """
    if answer is None:
        return ""

    lines = [f"## {answer.title}", "", answer.intro]

    for section in answer.sections:
        lines.extend(["", f"### {section.title}", "", section.content])

    lines.extend(["", answer.closing])

    return "\n".join(lines)


def render_persona(look: str, persona: Persona | None, narrator_prompt: str) -> str:
    """
    Собирает описание персонажа.

    Аргументы:
        look: описание облика с фотографии; пустая строка - фотографии не было.
        persona: рассказчик полями; None - рассказчик задан фразой.
        narrator_prompt: блок про рассказчика, ушедший в узел изложения.

    Возвращает:
        Облик и разложение персонажа в markdown. Пустую строку, если рассказчика
        нет.
    """
    lines: list[str] = []

    if look:
        lines.extend(["### Облик с фотографии", "", look])

    if persona is not None:
        lines.extend(
            [
                "",
                "### Рассказчик",
                "",
                f"- имя: {persona.name}",
                f"- пол: {persona.gender}",
                f"- характер: {persona.character}",
                f"- обращение к слушателю: {persona.address_to_listener}",
                f"- манера речи: {persona.speech_manner}",
                f"- любимые словечки: {', '.join(persona.favourite_words)}",
                f"- любимые звуки: {', '.join(persona.favourite_sounds)}",
                f"- отношение к предмету: {persona.attitude_to_subject}",
            ]
        )
    elif narrator_prompt:
        lines.extend(["", "### Рассказчик", "", narrator_prompt])

    return "\n".join(lines)


def render_notes(notes: ResearchNotes | None) -> str:
    """
    Собирает фактическую опору прогона.

    Аргументы:
        notes: опора, собранная по источникам; None - опоры ещё нет.

    Возвращает:
        Уверенность и список источников в markdown. Пустую строку, если опоры нет.
    """
    if notes is None:
        return ""

    lines = [f"уверенность: {notes.confidence}", "", "источники:"]
    lines.extend(f"- {url}" for url in notes.sources)

    return "\n".join(lines)


def call_argument(arguments: dict) -> str:
    """
    Сжимает аргументы вызова инструмента до одной строки.

    Имена аргументов не перечисляются: у каждого инструмента они свои, а в
    строке хода прогона важно само значение - запрос или адрес.

    Аргументы:
        arguments: аргументы вызова.

    Возвращает:
        Значения аргументов через запятую. Пустую строку, если аргументов нет.
    """
    return ", ".join(str(value) for value in arguments.values())


def stage_lines(name: str, step: ResearchStep | None) -> list[str]:
    """
    Превращает этап прогона в строки хода прогона.

    Аргументы:
        name: имя завершённого этапа.
        step: шаг графа, если этап - шаг ресёрча; иначе None.

    Возвращает:
        Для шага ресёрча - по строке на каждый объявленный вызов инструмента и на
        каждый его исход, плюс строку про собранные факты и про написанный текст;
        шаг без вызовов и без итога строк не даёт. Для остальных этапов - одну
        строку с именем этапа.
    """
    if step is None:
        return [name]

    lines = [
        f"вызов {tool_name}: {call_argument(arguments = arguments)}"
        for tool_name, arguments in step.tool_calls
    ]
    lines.extend(f"{tool_name} - {result}" for tool_name, result in step.tool_results)

    if step.notes is not None:
        lines.append("факты собраны")
    if step.answer is not None:
        lines.append("текст написан")

    return lines


def render_progress(lines: list[str]) -> str:
    """
    Собирает ход прогона.

    Аргументы:
        lines: пройденные этапы в порядке прохождения.

    Возвращает:
        Список этапов в markdown. Пустую строку, если этапов не было.
    """
    return "\n".join(f"- {line}" for line in lines)


def render_timing(timing: Stopwatch) -> str:
    """
    Собирает таблицу длительностей этапов.

    Аргументы:
        timing: копилка замеров.

    Возвращает:
        Таблицу в блоке кода. Пустую строку, если замеров не было.
    """
    table = timing.render_table()
    if not table:
        return ""

    return f"```\n{table}\n```"
