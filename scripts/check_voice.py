"""
Ручная проверка подбора голоса: картинка - облик - персонаж - настройки голоса.

Прогон идёт в три прохода: список голосов у модели синтеза, разбор картинок
vl-моделью, сборка персонажа и подбор голоса текстовой моделью. Так модели
меняются в памяти один раз, а не на каждой картинке.

Запуск:
    .venv/bin/python scripts/check_voice.py docs/persona_images/hulk/images4.jpeg

```
# по одной картинке из каждой папки
for folder in docs/persona_images/*/; do .venv/bin/python scripts/check_voice.py "$folder" --limit 1; done
```

"""

import argparse
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from assistant.integrations.llm.client import build_llm
from assistant.integrations.llm.profiles import NodeRole
from assistant.integrations.speaking import SpeechSynthesizer
from assistant.observability import setup_console_output
from assistant.persona import Persona, build_persona, describe_look, pick_voice
from assistant.variables import SPEAKING_CONFIG, VISION_MODEL, VISION_PROVIDER

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def collect_images(path: Path, limit: int) -> list[Path]:
    """
    Собирает список картинок по пути.

    Аргументы:
        path: файл с картинкой или каталог, который обходится вглубь.
        limit: сколько взять; ноль и меньше - все.

    Возвращает:
        Пути к картинкам по возрастанию имени.
    """
    if path.is_file():
        found = [path]
    else:
        found = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )

    return found[:limit] if limit > 0 else found


def collect_looks(llm: ChatOpenAI, images: list[Path]) -> dict[Path, str]:
    """
    Разбирает облик на каждой картинке.

    Аргументы:
        llm: клиент модели, принимающей картинки.
        images: пути к картинкам.

    Возвращает:
        Описания облика по путям к картинкам. Неудачные картинки в словарь
        не попадают.
    """
    looks: dict[Path, str] = {}

    for image_path in images:
        look, error = describe_look(llm = llm, image_path = image_path, callbacks = [])
        if error:
            print(f"{image_path}: {error}", flush = True)
            continue

        looks[image_path] = look

    return looks


def check_voice(
    writing_llm: ChatOpenAI,
    extraction_llm: ChatOpenAI,
    image_path: Path,
    look: str,
    speakers: list[str],
) -> None:
    """
    Строит персонажа по облику, подбирает ему голос и печатает результат.

    Аргументы:
        writing_llm: клиент модели для сборки персонажа.
        extraction_llm: клиент модели для подбора голоса.
        image_path: файл с картинкой; печатается для связи с результатом.
        look: описание облика персонажа.
        speakers: имена голосов, которые знает модель синтеза.

    Возвращает:
        Ничего.
    """
    persona, error = build_persona(llm = writing_llm, look = look, callbacks = [])
    if error:
        print(f"{image_path}: {error}", flush = True)
        return

    started = time.monotonic()
    settings, error = pick_voice(
        llm = extraction_llm,
        persona = persona,
        speakers = speakers,
        callbacks = [],
    )
    spent_seconds = time.monotonic() - started

    if error:
        print(f"{image_path}: {error}, {spent_seconds:.1f} с", flush = True)
        return

    print(f"{image_path}: {describe_persona(persona = persona)}", flush = True)
    print(
        f"    голос {settings.speaker}, темп {settings.rate}, высота {settings.pitch}, "
        f"эффект {settings.effect} ({settings.effect_strength}), {spent_seconds:.1f} с\n",
        flush = True,
    )


def describe_persona(persona: Persona) -> str:
    """
    Собирает строку с именем, полом и манерой речи персонажа.

    Аргументы:
        persona: рассказчик, выведенный из облика.

    Возвращает:
        Строку для печати.
    """
    return f"{persona.name} ({persona.gender}), манера: {persona.speech_manner}"


def main() -> None:
    """
    Прогоняет картинки по указанному пути через сборку персонажа и подбор голоса.

    Возвращает:
        Ничего.
    """
    setup_console_output()
    parser = argparse.ArgumentParser(description = "Проверка подбора голоса под персонажа")
    parser.add_argument("path", help = "файл с картинкой или каталог с картинками")
    parser.add_argument("--limit", type = int, default = 0, help = "сколько взять; ноль - все")
    arguments = parser.parse_args()

    synthesizer = SpeechSynthesizer(config = SPEAKING_CONFIG)
    speakers, error = synthesizer.available_speakers()
    if error:
        print(f"Голоса не получены: {error}")
        return

    print(f"--- голоса [{SPEAKING_CONFIG.model_id}]: {', '.join(speakers)}\n", flush = True)

    images = collect_images(path = Path(arguments.path), limit = arguments.limit)
    vision_llm = build_llm(
        role = NodeRole.VISION,
        is_reasoning_forced = False,
        model = VISION_MODEL,
        provider = VISION_PROVIDER,
    )
    print(f"--- облик: [{VISION_PROVIDER}] {VISION_MODEL}, картинок {len(images)}\n", flush = True)
    looks = collect_looks(llm = vision_llm, images = images)

    writing_llm = build_llm(role = NodeRole.WRITING, is_reasoning_forced = False, model = None)
    extraction_llm = build_llm(role = NodeRole.EXTRACTION, is_reasoning_forced = False, model = None)
    print(f"--- персонаж и голос: {writing_llm.model_name}, описаний {len(looks)}\n", flush = True)

    for image_path, look in looks.items():
        check_voice(
            writing_llm = writing_llm,
            extraction_llm = extraction_llm,
            image_path = image_path,
            look = look,
            speakers = speakers,
        )


if __name__ == "__main__":
    main()
