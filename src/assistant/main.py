"""
Точка входа: вопрос из командной строки, голосом или записью, ответ на экран.
Манеру изложения задаёт фотография персонажа либо фраза о рассказчике.
"""

import argparse
import sys
from pathlib import Path

from assistant.graph import (
    RESUMABLE_NODES,
    Answer,
    ResearchNotes,
    describe_nodes,
    list_runs,
    resume_research,
)
from assistant.integrations.listening import SILENCE_LEVEL, SpeechRecognizer, record
from assistant.observability import setup_console_output
from assistant.omni import run_omni_assistant
from assistant.persona import PersonaMode
from assistant.variables import LISTENING_CONFIG, LLM_PROVIDER, PERSONA_MODE

# Значение --record без числа: писать до нажатия Enter.
_RECORD_UNTIL_ENTER = 0.0


def build_parser() -> argparse.ArgumentParser:
    """
    Собирает разбор аргументов командной строки.

    Возвращает:
        Готовый разборщик.
    """
    parser = argparse.ArgumentParser(description = "Ресёрчер: поиск в интернете и ответ по источникам")
    parser.add_argument(
        "question",
        nargs = "?",
        help = "вопрос текстом",
    )
    parser.add_argument(
        "--audio",
        help = "файл с записью вопроса; формат любой",
    )
    parser.add_argument(
        "--image",
        help = "фотография персонажа, от лица которого ведётся экскурсия",
    )
    parser.add_argument(
        "--narrator",
        metavar = "ФРАЗА",
        help = "рассказчик фразой, например «угрюмый халк»; вместо --image",
    )
    parser.add_argument(
        "--persona-mode",
        choices = [mode.value for mode in PersonaMode],
        default = PERSONA_MODE,
        help = "как строить рассказчика по фотографии: фразой или полями схемы",
    )
    parser.add_argument(
        "--resume",
        metavar = "ПРОГОН",
        help = "переиграть записанный прогон; идентификатор совпадает с именем файла в logs/traces",
    )
    parser.add_argument(
        "--from",
        dest = "from_node",
        choices = RESUMABLE_NODES,
        help = "узел, с которого продолжить прогон; только вместе с --resume",
    )
    parser.add_argument(
        "--list-runs",
        action = "store_true",
        help = "показать прогоны, которые можно переиграть, и выйти",
    )
    parser.add_argument(
        "--record",
        nargs = "?",
        const = _RECORD_UNTIL_ENTER,
        type = float,
        metavar = "СЕКУНДЫ",
        help = "записать вопрос с микрофона; без числа - до нажатия Enter",
    )
    return parser


def resolve_question(arguments: argparse.Namespace) -> str:
    """
    Достаёт текст вопроса из аргументов: как есть, из файла или с микрофона.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Текст вопроса либо пустую строку, если получить его не вышло. Причина
        при этом уже напечатана.
    """
    sources = [arguments.question, arguments.audio, arguments.record]
    given = [source for source in sources if source is not None]

    if not given:
        print("Нужен вопрос: текстом, файлом --audio или записью --record.")
        return ""
    if len(given) > 1:
        print("Вопрос задаётся одним способом: текстом, --audio или --record.")
        return ""

    if arguments.question is not None:
        return arguments.question

    if arguments.audio is not None:
        return transcribe_file(audio_path = Path(arguments.audio))

    return transcribe_recording(seconds = arguments.record)


def transcribe_recording(seconds: float) -> str:
    """
    Пишет вопрос с микрофона и распознаёт его.

    Аргументы:
        seconds: сколько секунд писать; ноль и меньше - до нажатия Enter.

    Возвращает:
        Текст вопроса либо пустую строку, если запись или распознавание не
        удались.
    """
    outcome = record(
        seconds = None if seconds <= _RECORD_UNTIL_ENTER else seconds,
        config = LISTENING_CONFIG,
    )

    if outcome.error or outcome.path is None:
        print(f"Записать не вышло: {outcome.error}")
        return ""

    print(f"[запись] {outcome.path} - {outcome.seconds:.1f} с, громкость {outcome.peak_level:.2f}")
    if outcome.peak_level < SILENCE_LEVEL:
        print(
            "[запись] в файле тишина. На macOS проверьте разрешение на микрофон "
            "у терминала в настройках приватности и выбранное устройство ввода."
        )

    return transcribe_file(audio_path = outcome.path)


def transcribe_file(audio_path: Path) -> str:
    """
    Распознаёт запись из файла.

    Аргументы:
        audio_path: файл с записью вопроса.

    Возвращает:
        Текст вопроса либо пустую строку, если распознать не вышло.
    """
    recognizer = SpeechRecognizer(config = LISTENING_CONFIG)
    outcome = recognizer.transcribe(audio_path = audio_path)

    if outcome.error:
        print(f"Распознать не вышло: {outcome.error}")
        return ""

    source = "из кеша" if outcome.from_cache else "распознано"
    print(f"[вопрос] {source}: {outcome.text}")
    return outcome.text


def print_runs() -> None:
    """
    Печатает прогоны, которые можно переиграть.

    Возвращает:
        Ничего.
    """
    runs = list_runs()
    if not runs:
        print("Записанных прогонов нет.")
        return

    print("Прогоны, свежие первыми:")
    for run_id, question in runs:
        print(f"  {run_id}  {question}")


def print_answer(answer: Answer, notes: ResearchNotes) -> None:
    """
    Печатает итоговый текст и опору, на которой он построен.

    Аргументы:
        answer: итоговый текст.
        notes: фактическая опора.

    Возвращает:
        Ничего.
    """
    print(f"\n=== {answer.title} ===\n")
    print(answer.intro)

    for section in answer.sections:
        print(f"\n## {section.title}\n")
        print(section.content)

    print(f"\n{answer.closing}")

    print(f"\n--- опора ---\nуверенность: {notes.confidence}")
    print("источники:")
    for url in notes.sources:
        print(f"  - {url}")


def run_resume(arguments: argparse.Namespace) -> None:
    """
    Переигрывает записанный прогон с указанного узла.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Ничего. При неудаче завершает процесс кодом 1.
    """
    given = [arguments.question, arguments.audio, arguments.record, arguments.image]
    if any(source is not None for source in given):
        print("--resume не сочетается с вопросом, --audio, --record и --image.")
        sys.exit(1)

    if arguments.from_node is None:
        print("К --resume нужен --from: с какого узла продолжать.")
        sys.exit(1)

    answer, notes, error = resume_research(
        run_id = arguments.resume,
        from_node = arguments.from_node,
        narrator_prompt = arguments.narrator,
    )

    if error:
        print(f"Переиграть прогон не вышло: {error}")
        sys.exit(1)

    print_answer(answer = answer, notes = notes)


def main() -> None:
    """
    Разбирает аргументы командной строки и печатает ответ ресёрчера.

    Возвращает:
        Ничего. При неудаче с вопросом завершает процесс кодом 1.
    """
    setup_console_output()

    arguments = build_parser().parse_args()

    if arguments.list_runs:
        print_runs()
        return

    print(f"[провайдер] {LLM_PROVIDER}")
    for line in describe_nodes():
        print(f"  {line}")

    if arguments.resume:
        run_resume(arguments = arguments)
        return

    question = resolve_question(arguments = arguments)
    if not question:
        sys.exit(1)

    if arguments.image and arguments.narrator:
        print("Рассказчик задаётся одним способом: --image или --narrator.")
        sys.exit(1)

    outcome = run_omni_assistant(
        image_path = Path(arguments.image) if arguments.image else None,
        narrator_style = arguments.narrator,
        persona_mode = PersonaMode(arguments.persona_mode),
        question = question,
    )

    if outcome.error:
        print(f"Разобрать фотографию не вышло: {outcome.error}")
        sys.exit(1)

    print_answer(answer = outcome.answer, notes = outcome.notes)


if __name__ == "__main__":
    main()
