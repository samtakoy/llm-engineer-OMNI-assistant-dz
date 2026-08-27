"""
Точка входа: вопрос из командной строки, голосом или записью, ответ на экран.
"""

import argparse
import sys
from pathlib import Path

from assistant.graph.graph import describe_nodes, run_research
from assistant.integrations.llm.client import build_provider_config
from assistant.integrations.speech import SILENCE_LEVEL, SpeechRecognizer, record
from assistant.variables import LLM_PROVIDER, SPEECH_CONFIG

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
        config = SPEECH_CONFIG,
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
    recognizer = SpeechRecognizer(config = SPEECH_CONFIG)
    outcome = recognizer.transcribe(audio_path = audio_path)

    if outcome.error:
        print(f"Распознать не вышло: {outcome.error}")
        return ""

    source = "из кеша" if outcome.from_cache else "распознано"
    print(f"[вопрос] {source}: {outcome.text}")
    return outcome.text


def main() -> None:
    """
    Разбирает аргументы командной строки и печатает ответ ресёрчера.

    Возвращает:
        Ничего. При неудаче с вопросом завершает процесс кодом 1.
    """
    arguments = build_parser().parse_args()

    question = resolve_question(arguments = arguments)
    if not question:
        sys.exit(1)

    print(f"[модель] {build_provider_config(provider = LLM_PROVIDER).model}")
    for line in describe_nodes():
        print(f"  {line}")

    answer, notes = run_research(question = question)

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


if __name__ == "__main__":
    main()
