"""
Запись исхода прогона в markdown рядом с озвучкой.

Файлы падают в папку озвучки под тем же штампом времени, что и звук: запрос
пользователя в одном файле, текст экскурсии во втором, облик персонажа с
разложением рассказчика в третьем. Содержание повторяет то, что показывает
web-интерфейс.
"""

from pathlib import Path

from assistant.graph import Answer
from assistant.persona import Persona


def render_answer_document(answer: Answer) -> str:
    """
    Собирает текст экскурсии.

    Аргументы:
        answer: итоговый текст.

    Возвращает:
        Заголовок, вступление, разделы и завершение в markdown.
    """
    lines = [f"## {answer.title}", "", answer.intro]

    for section in answer.sections:
        lines.extend(["", f"### {section.title}", "", section.content])

    lines.extend(["", answer.closing])

    return "\n".join(lines)


def render_persona_document(look: str, persona: Persona | None, narrator_prompt: str) -> str:
    """
    Собирает описание персонажа.

    Аргументы:
        look: описание облика с фотографии; пустая строка - фотографии не было.
        persona: рассказчик полями; None - рассказчик задан фразой.
        narrator_prompt: блок про рассказчика, ушедший в узел изложения.

    Возвращает:
        Облик и разложение персонажа в markdown. Пустую строку, если нет ни
        облика, ни рассказчика.
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

    return "\n".join(lines).strip()


def write_document(path: Path, text: str) -> Path:
    """
    Кладёт markdown в файл, заводя папку под него.

    Аргументы:
        path: файл, в который писать.
        text: содержание файла.

    Возвращает:
        Файл, в который лёг текст.
    """
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(f"{text}\n", encoding = "utf-8")

    return path


def write_run_documents(
    question: str,
    answer: Answer,
    look: str,
    persona: Persona | None,
    narrator_prompt: str,
    directory: Path,
    stamp: str,
) -> list[Path]:
    """
    Пишет markdown прогона: запрос пользователя, текст экскурсии и описание
    персонажа.

    Файл запроса не заводится с пустым запросом, файл персонажа - когда нет ни
    облика, ни рассказчика.

    Аргументы:
        question: запрос пользователя, с которым прошёл прогон.
        answer: итоговый текст.
        look: описание облика с фотографии; пустая строка - фотографии не было.
        persona: рассказчик полями; None - рассказчик задан фразой.
        narrator_prompt: блок про рассказчика, ушедший в узел изложения.
        directory: папка, в которую писать.
        stamp: штамп времени прогона, общий со звуковыми файлами.

    Возвращает:
        Записанные файлы в порядке записи.
    """
    paths: list[Path] = []

    if question.strip():
        paths.append(
            write_document(
                path = directory / f"{stamp}-request.md",
                text = question.strip(),
            )
        )

    paths.append(
        write_document(
            path = directory / f"{stamp}-text.md",
            text = render_answer_document(answer = answer),
        )
    )

    persona_document = render_persona_document(
        look = look,
        persona = persona,
        narrator_prompt = narrator_prompt,
    )
    if persona_document:
        paths.append(
            write_document(
                path = directory / f"{stamp}-persona.md",
                text = persona_document,
            )
        )

    return paths
