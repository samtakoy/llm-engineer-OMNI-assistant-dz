"""
Ручная проверка синтеза: строка уходит в silero, звук ложится в файл.

Запуск:
    # какие голоса знает версия модели
    .venv/bin/python scripts/check_speaking.py --list

    # одна строка одним голосом
    .venv/bin/python scripts/check_speaking.py "Здравствуйте, я ваш экскурсовод" --speaker xenia

    # тот же текст всеми голосами версии
    .venv/bin/python scripts/check_speaking.py "Здравствуйте" --all-speakers

    # темп и высота разметкой ssml
    .venv/bin/python scripts/check_speaking.py "Здравствуйте" --speaker eugene --rate slow --pitch x-low
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from assistant.integrations.speaking import SpeechSynthesizer, VoiceSettings
from assistant.variables import SPEAKING_CONFIG, SPOKEN_PATH

RATE_VALUES = ("x-slow", "slow", "medium", "fast", "x-fast")
PITCH_VALUES = ("x-low", "low", "medium", "high", "x-high")


def output_path(speaker: str) -> Path:
    """
    Составляет путь к файлу с озвучкой.

    Аргументы:
        speaker: имя голоса; попадает в имя файла, чтобы озвучки разными
            голосами не затирали друг друга.

    Возвращает:
        Путь к файлу.
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return SPOKEN_PATH / f"{stamp}-{speaker}.wav"


def check_speaker(
    synthesizer: SpeechSynthesizer,
    text: str,
    speaker: str,
    rate: str,
    pitch: str,
) -> None:
    """
    Озвучивает текст одним голосом и печатает результат.

    Аргументы:
        synthesizer: синтезатор речи.
        text: что произнести.
        speaker: имя голоса.
        rate: темп речи.
        pitch: высота голоса.

    Возвращает:
        Ничего.
    """
    settings = VoiceSettings(speaker = speaker, rate = rate, pitch = pitch)
    started = time.monotonic()

    outcome = synthesizer.synthesize(
        text = text,
        settings = settings,
        output_path = output_path(speaker = speaker),
    )
    spent_seconds = time.monotonic() - started

    if outcome.error:
        print(f"{speaker}: {outcome.error}, {spent_seconds:.1f} с", flush = True)
        return

    print(
        f"{speaker}: {outcome.path}, звучание {outcome.seconds:.1f} с, "
        f"синтез {spent_seconds:.1f} с",
        flush = True,
    )


def main() -> None:
    """
    Прогоняет строку через синтез одним голосом либо всеми голосами версии.

    Возвращает:
        Ничего.
    """
    parser = argparse.ArgumentParser(description = "Проверка синтеза речи на строке")
    parser.add_argument("text", nargs = "?", help = "текст для озвучки")
    parser.add_argument("--speaker", default = "", help = "имя голоса из версии модели")
    parser.add_argument("--rate", choices = RATE_VALUES, default = "medium", help = "темп речи")
    parser.add_argument("--pitch", choices = PITCH_VALUES, default = "medium", help = "высота голоса")
    parser.add_argument("--list", action = "store_true", help = "показать голоса версии и выйти")
    parser.add_argument(
        "--all-speakers",
        action = "store_true",
        help = "озвучить текст каждым голосом версии",
    )
    arguments = parser.parse_args()

    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)
    speakers, error = synthesizer.available_speakers()
    if error:
        print(f"Голоса не получены: {error}")
        return

    print(f"[{SPEAKING_CONFIG.model_id}] голоса: {', '.join(speakers)}")
    if arguments.list:
        return

    if not arguments.text:
        print("Нужен текст для озвучки.")
        return

    chosen = speakers if arguments.all_speakers else [arguments.speaker or speakers[0]]
    for speaker in chosen:
        check_speaker(
            synthesizer = synthesizer,
            text = arguments.text,
            speaker = speaker,
            rate = arguments.rate,
            pitch = arguments.pitch,
        )


if __name__ == "__main__":
    main()
