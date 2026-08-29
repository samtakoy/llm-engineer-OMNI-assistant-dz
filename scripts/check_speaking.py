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
    .venv/bin/python scripts/check_speaking.py "Здравствуйте" --speaker eugene --rate slow --pitch low

    # тот же голос всеми эффектами реестра
    .venv/bin/python scripts/check_speaking.py "Здравствуйте" --speaker eugene --all-effects
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from assistant.integrations.speaking import (
    NO_EFFECT,
    SpeechSynthesizer,
    VoiceSettings,
    available_effects,
    pitch_values,
    rate_values,
    strength_values,
)
from assistant.observability import setup_console_output
from assistant.variables import SPEAKING_CONFIG, SPOKEN_PATH

RATE_VALUES = rate_values()
PITCH_VALUES = pitch_values()
STRENGTH_VALUES = strength_values()


def output_path(speaker: str, effect: str) -> Path:
    """
    Составляет путь к файлу с озвучкой.

    Аргументы:
        speaker: имя голоса; попадает в имя файла, чтобы озвучки разными
            голосами не затирали друг друга.
        effect: имя эффекта; попадает в имя файла по той же причине.

    Возвращает:
        Путь к файлу.
    """
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return SPOKEN_PATH / f"{stamp}-{speaker}-{effect}.wav"


def check_speaker(
    synthesizer: SpeechSynthesizer,
    text: str,
    speaker: str,
    rate: str,
    pitch: str,
    effect: str,
    effect_strength: str,
) -> None:
    """
    Озвучивает текст одним голосом с одним эффектом и печатает результат.

    Аргументы:
        synthesizer: синтезатор речи.
        text: что произнести.
        speaker: имя голоса.
        rate: темп речи.
        pitch: высота голоса.
        effect: имя звукового эффекта.
        effect_strength: сила эффекта.

    Возвращает:
        Ничего.
    """
    settings = VoiceSettings(
        speaker = speaker,
        rate = rate,
        pitch = pitch,
        effect = effect,
        effect_strength = effect_strength,
    )
    started = time.monotonic()

    outcome = synthesizer.synthesize(
        text = text,
        settings = settings,
        output_path = output_path(speaker = speaker, effect = effect),
    )
    spent_seconds = time.monotonic() - started

    if outcome.error:
        print(f"{speaker}/{effect}: {outcome.error}, {spent_seconds:.1f} с", flush = True)
        return

    print(
        f"{speaker}/{effect}: {outcome.path}, звучание {outcome.seconds:.1f} с, "
        f"синтез {spent_seconds:.1f} с",
        flush = True,
    )


def main() -> None:
    """
    Прогоняет строку через синтез: один голос или все, один эффект или все.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка синтеза речи на строке")
    parser.add_argument("text", nargs = "?", help = "текст для озвучки")
    parser.add_argument("--speaker", default = "", help = "имя голоса из версии модели")
    parser.add_argument("--rate", choices = RATE_VALUES, default = "medium", help = "темп речи")
    parser.add_argument("--pitch", choices = PITCH_VALUES, default = "medium", help = "высота голоса")
    parser.add_argument("--list", action = "store_true", help = "показать голоса версии и выйти")
    parser.add_argument(
        "--effect",
        choices = available_effects(),
        default = NO_EFFECT,
        help = "звуковой эффект поверх синтеза",
    )
    parser.add_argument(
        "--effect-strength",
        choices = STRENGTH_VALUES,
        default = "medium",
        help = "сила звукового эффекта",
    )
    parser.add_argument(
        "--all-effects",
        action = "store_true",
        help = "озвучить текст каждым эффектом реестра",
    )
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

    chosen_speakers = speakers if arguments.all_speakers else [arguments.speaker or speakers[0]]
    chosen_effects = available_effects() if arguments.all_effects else [arguments.effect]

    for speaker in chosen_speakers:
        for effect in chosen_effects:
            check_speaker(
                synthesizer = synthesizer,
                text = arguments.text,
                speaker = speaker,
                rate = arguments.rate,
                pitch = arguments.pitch,
                effect = effect,
                effect_strength = arguments.effect_strength,
            )


if __name__ == "__main__":
    main()
