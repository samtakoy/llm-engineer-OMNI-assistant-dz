"""
Ручная проверка разметки: персонаж с картинки размечает фразу и произносит её.

Прогон озвучивает фразу дважды одним и тем же голосом: без разметки и с
разметкой от модели. Разницу слышно при сравнении двух файлов.

Запуск:
    .venv/bin/python scripts/check_markup.py docs/persona_images/hulk/images.jpeg \
        "Этот зал помнит первый паровой молот"
"""

import argparse
from pathlib import Path

from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.integrations.speaking import SpeechSynthesizer, VoiceSettings, sanitize_markup
from assistant.persona import build_persona, describe_look, mark_up_speech, pick_voice
from assistant.variables import SPEAKING_CONFIG, SPOKEN_PATH, VISION_MODEL, VISION_PROVIDER


def speak(
    synthesizer: SpeechSynthesizer,
    text: str,
    settings: VoiceSettings,
    name: str,
) -> None:
    """
    Озвучивает текст и печатает результат.

    Аргументы:
        synthesizer: синтезатор речи.
        text: что произнести, с разметкой или без неё.
        settings: настройки голоса.
        name: имя прогона; попадает в имя файла и в строку вывода.

    Возвращает:
        Ничего.
    """
    outcome = synthesizer.synthesize(
        text = text,
        settings = settings,
        output_path = SPOKEN_PATH / f"markup-{name}-{settings.speaker}.wav",
    )

    if outcome.error:
        print(f"{name}: {outcome.error}", flush = True)
        return

    print(f"{name}: {outcome.path}, звучание {outcome.seconds:.1f} с", flush = True)


def main() -> None:
    """
    Строит персонажа по картинке, размечает фразу и озвучивает её дважды.

    Возвращает:
        Ничего.
    """
    parser = argparse.ArgumentParser(description = "Проверка разметки речи персонажем")
    parser.add_argument("image", help = "фотография персонажа")
    parser.add_argument("text", help = "фраза для озвучки")
    arguments = parser.parse_args()

    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)
    speakers, error = synthesizer.available_speakers()
    if error:
        print(f"Голоса не получены: {error}")
        return

    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = False,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )
    look, error = describe_look(llm = vision_llm, image_path = Path(arguments.image))
    if error:
        print(f"Облик не разобран: {error}")
        return

    writing_llm = build_llm(role = NodeRole.WRITING, is_reasoning_forced = False, model = None)
    persona, error = build_persona(llm = writing_llm, look = look)
    if error:
        print(f"Персонаж не собран: {error}")
        return

    print(f"[персонаж] {persona.name} ({persona.gender}), манера: {persona.speech_manner}\n")

    extraction_llm = build_llm(role = NodeRole.EXTRACTION, is_reasoning_forced = False, model = None)
    settings, error = pick_voice(llm = extraction_llm, persona = persona, speakers = speakers)
    if error:
        print(f"Голос не подобран: {error}")
        return

    print(
        f"[голос] {settings.speaker}, темп {settings.rate}, высота {settings.pitch}, "
        f"эффект {settings.effect} ({settings.effect_strength})\n"
    )

    marked, error = mark_up_speech(llm = writing_llm, persona = persona, text = arguments.text)
    if error:
        print(f"Разметка не получена: {error}")
        return

    print(f"[разметка]\n{marked}\n")

    body, has_markup = sanitize_markup(text = marked)
    print(f"[после чистки] теги остались: {has_markup}\n{body}\n")

    speak(synthesizer = synthesizer, text = arguments.text, settings = settings, name = "plain")
    speak(synthesizer = synthesizer, text = marked, settings = settings, name = "marked")


if __name__ == "__main__":
    main()
