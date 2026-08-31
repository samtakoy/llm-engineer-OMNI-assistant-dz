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
    latest_run_id,
    list_runs,
)
from assistant.observability import setup_console_output
from assistant.omni import (
    RECORD_UNTIL_ENTER,
    OmniOutcome,
    resume_omni_assistant,
    run_omni_assistant,
)
from assistant.persona import PersonaMode
from assistant.variables import LLM_PROVIDER, PERSONA_MODE

# Значение --reuse-facts, при котором берётся свежий записанный прогон.
LATEST_RUN = ""


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
        "--speak",
        action = "store_true",
        help = "озвучить готовый текст голосом персонажа",
    )
    parser.add_argument(
        "--markup",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "перед озвучкой размечать текст паузами и ударениями; только с --speak",
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
        "--reuse-facts",
        nargs = "?",
        const = LATEST_RUN,
        metavar = "ПРОГОН",
        help = "взять факты записанного прогона и сразу изложить их; без значения - свежий прогон",
    )
    parser.add_argument(
        "--list-runs",
        action = "store_true",
        help = "показать прогоны, которые можно переиграть, и выйти",
    )
    parser.add_argument(
        "--record",
        nargs = "?",
        const = RECORD_UNTIL_ENTER,
        type = float,
        metavar = "СЕКУНДЫ",
        help = "записать вопрос с микрофона; без числа - до нажатия Enter",
    )
    return parser


def check_question_sources(arguments: argparse.Namespace) -> str:
    """
    Проверяет, что вопрос задан ровно одним источником.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Причину отказа либо пустую строку, если источник ровно один.
    """
    sources = [arguments.question, arguments.audio, arguments.record]
    given = [source for source in sources if source is not None]

    if arguments.reuse_facts is not None:
        if given:
            return (
                "--reuse-facts не сочетается с вопросом, --audio и --record: "
                "вопрос берётся из записанного прогона."
            )
        return ""

    if not given:
        return "Нужен вопрос: текстом, файлом --audio или записью --record."
    if len(given) > 1:
        return "Вопрос задаётся одним способом: текстом, --audio или --record."

    return ""


def check_resume_arguments(arguments: argparse.Namespace) -> str:
    """
    Проверяет аргументы продолжения записанного прогона.

    Аргументы:
        arguments: разобранные аргументы командной строки.

    Возвращает:
        Причину отказа либо пустую строку, если аргументы сходятся.
    """
    given = [arguments.question, arguments.audio, arguments.record, arguments.image]
    if any(source is not None for source in given):
        return "--resume не сочетается с вопросом, --audio, --record и --image."

    if arguments.reuse_facts is not None:
        return "--resume не сочетается с --reuse-facts: это разные способы переиграть прогон."

    if arguments.from_node is None:
        return "К --resume нужен --from: с какого узла продолжать."

    return ""


def resolve_reuse_run_id(reuse_facts: str | None) -> tuple[str | None, str]:
    """
    Выбирает прогон, факты которого берутся готовыми.

    Аргументы:
        reuse_facts: значение --reuse-facts; LATEST_RUN - свежий записанный
            прогон, None - факты собираются заново.

    Возвращает:
        Пару «идентификатор прогона, причина отказа». Идентификатор равен None,
        если факты собираются заново.
    """
    if reuse_facts is None:
        return None, ""

    if reuse_facts != LATEST_RUN:
        return reuse_facts, ""

    run_id = latest_run_id()
    if not run_id:
        return None, "Записанных прогонов нет: факты брать неоткуда."

    return run_id, ""


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


def print_timing(outcome: OmniOutcome) -> None:
    """
    Печатает файлы с озвучкой, таблицу длительностей и путь к журналу прогона.

    Аргументы:
        outcome: исход прогона.

    Возвращает:
        Ничего.
    """
    if outcome.audio_paths:
        print("\n--- озвучка ---")
        for path in outcome.audio_paths:
            print(f"  {path}")

    table = outcome.timing.render_table()
    if table:
        print(f"\n--- длительности ---\n{table}")

    if outcome.trace_path is not None:
        print(f"\n--- журнал прогона ---\n  {outcome.trace_path}")


def main() -> None:
    """
    Разбирает аргументы командной строки и печатает ответ ресёрчера.

    Возвращает:
        Ничего. При неудаче завершает процесс кодом 1.
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
        refusal = check_resume_arguments(arguments = arguments)
        if refusal:
            print(refusal)
            sys.exit(1)

        outcome = resume_omni_assistant(
            resume_run_id = arguments.resume,
            from_node = arguments.from_node,
            narrator_style = arguments.narrator,
            is_speech_on = arguments.speak,
            is_markup_on = arguments.markup,
        )
    else:
        refusal = check_question_sources(arguments = arguments)
        if refusal:
            print(refusal)
            sys.exit(1)

        if arguments.image and arguments.narrator:
            print("Рассказчик задаётся одним способом: --image или --narrator.")
            sys.exit(1)

        reuse_run_id, refusal = resolve_reuse_run_id(reuse_facts = arguments.reuse_facts)
        if refusal:
            print(refusal)
            sys.exit(1)

        outcome = run_omni_assistant(
            image_path = Path(arguments.image) if arguments.image else None,
            narrator_style = arguments.narrator,
            persona_mode = PersonaMode(arguments.persona_mode),
            question_text = arguments.question,
            audio_path = Path(arguments.audio) if arguments.audio else None,
            record_seconds = arguments.record,
            reuse_run_id = reuse_run_id,
            is_speech_on = arguments.speak,
            is_markup_on = arguments.markup,
        )

    if outcome.error:
        print(f"Прогон не удался: {outcome.error}")
        sys.exit(1)

    print_answer(answer = outcome.answer, notes = outcome.notes)
    print_timing(outcome = outcome)


if __name__ == "__main__":
    main()
