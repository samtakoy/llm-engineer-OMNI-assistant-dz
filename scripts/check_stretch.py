"""
Ручная проверка растяжения слов: одна фраза озвучивается несколькими формами
записи тянутого слова.

Прогон печатает длительность каждого варианта. Растяжение, которое синтез
слышит, даёт секунды больше эталона; проглоченное растяжение оставляет
длительность прежней. Формы записи собираются генератором из слова и буквы,
списка вариантов в коде нет.

Запуск:
    .venv/bin/python scripts/check_stretch.py "ну вот и всё" ну у --speaker aidar --repeats 3
"""

import argparse

from assistant.integrations.speaking import NO_EFFECT, SpeechSynthesizer, VoiceSettings
from assistant.observability import setup_console_output
from assistant.variables import SPEAKING_CONFIG, SPOKEN_PATH


def build_variants(word: str, vowel: str, repeats: int) -> list[tuple[str, str]]:
    """
    Собирает формы записи тянутого слова.

    Гласная размножается двумя способами: слитно и через дефис. Число копий
    растёт от одной до заданного предела.

    Аргументы:
        word: слово, которое тянем.
        vowel: буква, которую размножаем.
        repeats: наибольшее число добавленных копий буквы.

    Возвращает:
        Пары «имя варианта, слово». Пустой список, если буквы в слове нет.
    """
    position = word.find(vowel)
    if position < 0:
        return []

    head = word[:position + 1]
    tail = word[position + 1:]

    variants: list[tuple[str, str]] = []
    for count in range(1, repeats + 1):
        variants.append((f"слитно-{count + 1}", f"{head}{vowel * count}{tail}"))
        variants.append((f"дефис-{count + 1}", f"{head}{f'-{vowel}' * count}{tail}"))

    return variants


def speak_variant(
    synthesizer: SpeechSynthesizer,
    text: str,
    settings: VoiceSettings,
    name: str,
) -> float:
    """
    Озвучивает вариант фразы и печатает строку отчёта.

    Аргументы:
        synthesizer: синтезатор речи.
        text: фраза целиком.
        settings: настройки голоса.
        name: имя варианта; попадает в имя файла и в строку отчёта.

    Возвращает:
        Длительность звучания в секундах, ноль при неудаче.
    """
    outcome = synthesizer.synthesize(
        text = text,
        settings = settings,
        output_path = SPOKEN_PATH / f"stretch-{name}-{settings.speaker}.wav",
    )

    if outcome.error:
        print(f"{name:<12} {outcome.error}", flush = True)
        return 0.0

    print(f"{name:<12} {outcome.seconds:5.2f} с  {text}", flush = True)
    return outcome.seconds


def main() -> None:
    """
    Озвучивает фразу эталоном и всеми формами тянутого слова.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка растяжения слов синтезом")
    parser.add_argument("text", help = "фраза для озвучки")
    parser.add_argument("word", help = "слово из фразы, которое тянем")
    parser.add_argument("vowel", help = "буква, которую размножаем")
    parser.add_argument("--speaker", required = True, help = "имя голоса модели синтеза")
    parser.add_argument(
        "--repeats",
        required = True,
        type = int,
        help = "наибольшее число добавленных копий буквы",
    )
    arguments = parser.parse_args()

    if arguments.word not in arguments.text:
        print(f"Слова {arguments.word} во фразе нет")
        return

    variants = build_variants(
        word = arguments.word,
        vowel = arguments.vowel,
        repeats = arguments.repeats,
    )
    if not variants:
        print(f"Буквы {arguments.vowel} в слове {arguments.word} нет")
        return

    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)
    speakers, error = synthesizer.available_speakers()
    if error:
        print(f"Голоса не получены: {error}")
        return

    if arguments.speaker not in speakers:
        print(f"Голоса {arguments.speaker} нет в модели, доступны: {', '.join(speakers)}")
        return

    settings = VoiceSettings(
        speaker = arguments.speaker,
        rate = "medium",
        pitch = "medium",
        effect = NO_EFFECT,
        effect_strength = "low",
    )

    print(f"[фраза] {arguments.text}")
    print(f"[слово] {arguments.word}, буква {arguments.vowel}\n")

    reference = speak_variant(
        synthesizer = synthesizer,
        text = arguments.text,
        settings = settings,
        name = "эталон",
    )

    for name, stretched_word in variants:
        speak_variant(
            synthesizer = synthesizer,
            text = arguments.text.replace(arguments.word, stretched_word),
            settings = settings,
            name = name,
        )

    print(f"\nЭталон {reference:.2f} с. Растяжение слышно там, где секунд заметно больше.")
    print(f"Файлы: {SPOKEN_PATH}")


if __name__ == "__main__":
    main()
